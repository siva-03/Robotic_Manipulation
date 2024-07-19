import logging
from copy import copy
from enum import Enum

import numpy as np
from pydrake.all import (
    AbstractValue,
    AddMultibodyPlantSceneGraph,
    Concatenate,
    DiagramBuilder,
    InputPortIndex,
    LeafSystem,
    MeshcatVisualizer,
    Parser,
    PiecewisePolynomial,
    PiecewisePose,
    PointCloud,
    PortSwitch,
    RandomGenerator,
    RigidTransform,
    RollPitchYaw,
    Simulator,
    StartMeshcat,
    UniformlyRandomRotationMatrix,
    GetScopedFrameByName,
    DepthImageToPointCloud,
    MeshcatPointCloudVisualizer,
)

from manipulation import ConfigureParser, FindResource, running_as_notebook
from manipulation.clutter import GenerateAntipodalGraspCandidate
from manipulation.meshcat_utils import AddMeshcatTriad, StopButton
from manipulation.pick import (
    MakeGripperCommandTrajectory,
    MakeGripperFrames,
    MakeGripperPoseTrajectory,
)
from manipulation.scenarios import AddIiwaDifferentialIK, ycb
from manipulation.station import (
    AddPointClouds,
    AppendDirectives,
    LoadScenario,
    MakeHardwareStation,
)
from manipulation.systems import ExtractPose

class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Differential IK")


logging.getLogger("drake").addFilter(NoDiffIKWarnings())

running_as_notebook = True

# Start the visualizer.
meshcat = StartMeshcat()

rng = np.random.default_rng(np.random.randint(10, 200))  # this is for python 135
generator = RandomGenerator(rng.integers(0, 1000))  # this is for c++


# Another diagram for the objects the robot "knows about": gripper, cameras, bins.  Think of this as the model in the robot's head.
def make_internal_model():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/clutter_planning.dmd.yaml")
    plant.Finalize()
    return builder.Build()


# Takes 3 point clouds (in world coordinates) as input, and outputs and estimated pose for the mustard bottle.
class GraspSelector(LeafSystem):
    def __init__(self, plant, bin_instance, camera_body_indices):
        LeafSystem.__init__(self)
        model_point_cloud = AbstractValue.Make(PointCloud(0))
        self.DeclareAbstractInputPort("cloud0_W", model_point_cloud)
        self.DeclareAbstractInputPort("cloud1_W", model_point_cloud)
        self.DeclareAbstractInputPort("cloud2_W", model_point_cloud)
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )

        port = self.DeclareAbstractOutputPort(
            "grasp_selection",
            lambda: AbstractValue.Make((np.inf, RigidTransform())),
            self.SelectGrasp,
        )
        port.disable_caching_by_default()

        # Compute crop box.
        context = plant.CreateDefaultContext()
        bin_body = plant.GetBodyByName("bin_base", bin_instance)
        X_B = plant.EvalBodyPoseInWorld(context, bin_body)
        margin = 0.001  # only because simulation is perfect!
        a = X_B.multiply(
            [-0.22 + 0.025 + margin, -0.29 + 0.025 + margin, 0.015 + margin]
        )
        b = X_B.multiply([0.22 - 0.1 - margin, 0.29 - 0.025 - margin, 2.0])
        self._crop_lower = np.minimum(a, b)
        self._crop_upper = np.maximum(a, b)

        self._internal_model = make_internal_model()
        self._internal_model_context = self._internal_model.CreateDefaultContext()
        self._rng = np.random.default_rng()
        self._camera_body_indices = camera_body_indices

    def SelectGrasp(self, context, output):
        body_poses = self.get_input_port(3).Eval(context)
        pcd = []
        for i in range(3):
            cloud = self.get_input_port(i).Eval(context)
            pcd.append(cloud.Crop(self._crop_lower, self._crop_upper))
            pcd[i].EstimateNormals(radius=0.1, num_closest=30)

            # Flip normals toward camera
            X_WC = body_poses[self._camera_body_indices[i]]
            pcd[i].FlipNormalsTowardPoint(X_WC.translation())
        merged_pcd = Concatenate(pcd)
        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)

        costs = []
        X_Gs = []
        # TODO(russt): Take the randomness from an input port, and re-enable
        # caching.
        for i in range(100 if running_as_notebook else 100):
            cost, X_G = GenerateAntipodalGraspCandidate(
                self._internal_model,
                self._internal_model_context,
                down_sampled_pcd,
                self._rng,
            )
            if np.isfinite(cost):
                costs.append(cost)
                X_Gs.append(X_G)

        if len(costs) == 0:
            # Didn't find a viable grasp candidate
            X_WG = RigidTransform(
                RollPitchYaw(-np.pi / 2, 0, np.pi / 2), [0.5, 0, 0.22]
            )
            output.set_value((np.inf, X_WG))
        else:
            best = np.argmin(costs)
            output.set_value((costs[best], X_Gs[best]))


class PlannerState(Enum):
    WAIT_FOR_OBJECTS_TO_SETTLE = 1
    PICKING_FROM_X_BIN = 2
    # PICKING_FROM_Y_BIN = 3
    GO_HOME = 4


class Planner(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._gripper_body_index = plant.GetBodyByName("body").index()
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self._x_bin_grasp_index = self.DeclareAbstractInputPort(
            "x_bin_grasp", AbstractValue.Make((np.inf, RigidTransform()))
        ).get_index()
        # self._y_bin_grasp_index = self.DeclareAbstractInputPort(
        #     "y_bin_grasp", AbstractValue.Make((np.inf, RigidTransform()))
        # ).get_index()
        self._wsg_state_index = self.DeclareVectorInputPort("wsg_state", 2).get_index()

        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE)
        )
        self._traj_X_G_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePose())
        )
        self._traj_wsg_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self._times_index = self.DeclareAbstractState(
            AbstractValue.Make({"initial": 0.0})
        )
        self._attempts_index = self.DeclareDiscreteState(1)

        self.DeclareAbstractOutputPort(
            "X_WG",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose,
        )
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)

        # For GoHome mode.
        num_positions = 7
        self._iiwa_position_index = self.DeclareVectorInputPort(
            "iiwa_position", num_positions
        ).get_index()
        self.DeclareAbstractOutputPort(
            "control_mode",
            lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode,
        )
        self.DeclareAbstractOutputPort(
            "reset_diff_ik",
            lambda: AbstractValue.Make(False),
            self.CalcDiffIKReset,
        )
        self._q0_index = self.DeclareDiscreteState(num_positions)  # for q0
        self._traj_q_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self.DeclareVectorOutputPort(
            "iiwa_position_command", num_positions, self.CalcIiwaPosition
        )
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

    def Update(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        current_time = context.get_time()
        times = context.get_abstract_state(int(self._times_index)).get_value()

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            if context.get_time() - times["initial"] > 1.0:
                self.Plan(context, state)
            return
        elif mode == PlannerState.GO_HOME:
            traj_q = context.get_mutable_abstract_state(
                int(self._traj_q_index)
            ).get_value()
            if not traj_q.is_time_in_range(current_time):
                self.Plan(context, state)
            return

        # If we are between pick and place and the gripper is closed, then
        # we've missed or dropped the object.  Time to replan.
        if current_time > times["postpick"] and current_time < times["preplace"]:
            wsg_state = self.get_input_port(self._wsg_state_index).Eval(context)
            if wsg_state[0] < 0.01:
                attempts = state.get_mutable_discrete_state(
                    int(self._attempts_index)
                ).get_mutable_value()
                # if attempts[0] > 5:
                #     # If I've failed 5 times in a row, then switch bins.
                #     print(
                #         "Switching to the other bin after 5 consecutive failed attempts"
                #     )
                #     attempts[0] = 0
                #     if mode == PlannerState.PICKING_FROM_X_BIN:
                #         state.get_mutable_abstract_state(
                #             int(self._mode_index)
                #         ).set_value(PlannerState.PICKING_FROM_Y_BIN)
                #     else:
                #         state.get_mutable_abstract_state(
                #             int(self._mode_index)
                #         ).set_value(PlannerState.PICKING_FROM_X_BIN)
                #     self.Plan(context, state)
                #     return

                attempts[0] += 1
                print("In dropped item mode")
                state.get_mutable_abstract_state(int(self._mode_index)).set_value(
                    PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE
                )
                times = {"initial": current_time}
                state.get_mutable_abstract_state(int(self._times_index)).set_value(
                    times
                )
                X_G = self.get_input_port(0).Eval(context)[
                    int(self._gripper_body_index)
                ]
                state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
                    PiecewisePose.MakeLinear([current_time, np.inf], [X_G, X_G])
                )
                return

        traj_X_G = context.get_abstract_state(int(self._traj_X_G_index)).get_value()
        if not traj_X_G.is_time_in_range(current_time):
            self.Plan(context, state)
            return

        X_G = self.get_input_port(0).Eval(context)[int(self._gripper_body_index)]
        # if current_time > 10 and current_time < 12:
        #    self.GoHome(context, state)
        #    return
        if (
            np.linalg.norm(
                traj_X_G.GetPose(current_time).translation() - X_G.translation()
            )
            > 0.2
        ):
            # If my trajectory tracking has gone this wrong, then I'd better
            # stop and replan.  TODO(russt): Go home, in joint coordinates,
            # instead.
            self.GoHome(context, state)
            return

    def GoHome(self, context, state):
        print("Replanning due to large tracking error.")
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(
            PlannerState.GO_HOME
        )
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q0 = copy(context.get_discrete_state(self._q0_index).get_value())
        q0[0] = q[0]  # Safer to not reset the first joint.

        current_time = context.get_time()
        q_traj = PiecewisePolynomial.FirstOrderHold(
            [current_time, current_time + 5.0], np.vstack((q, q0)).T
        )
        state.get_mutable_abstract_state(int(self._traj_q_index)).set_value(q_traj)

    def Plan(self, context, state):
        mode = copy(state.get_mutable_abstract_state(int(self._mode_index)).get_value())

        X_G = {
            "initial": self.get_input_port(0).Eval(context)[
                int(self._gripper_body_index)
            ]
        }

        cost = np.inf
        for i in range(10):
            # if mode == PlannerState.PICKING_FROM_Y_BIN:
            #     cost, X_G["pick"] = self.get_input_port(self._y_bin_grasp_index).Eval(
            #         context
            #     )
            #     if np.isinf(cost):
            #         mode = PlannerState.PICKING_FROM_X_BIN
            # else:
            #     cost, X_G["pick"] = self.get_input_port(self._x_bin_grasp_index).Eval(
            #         context
            #     )
            #     if np.isinf(cost):
            #         mode = PlannerState.PICKING_FROM_Y_BIN
            #     else:
            #         mode = PlannerState.PICKING_FROM_X_BIN
            cost, X_G["pick"] = self.get_input_port(self._x_bin_grasp_index).Eval(
                    context
                )
            if not np.isinf(cost):
                mode = PlannerState.PICKING_FROM_X_BIN
                break
        # print(mode)
        assert not np.isinf(
            cost
        ), "Could not find a valid grasp in either bin after 10 attempts"
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(mode)

        # TODO(russt): The randomness should come in through a random input
        # port.
        if mode == PlannerState.PICKING_FROM_X_BIN:
            # Place in Y bin:
            X_G["place"] = RigidTransform(
                RollPitchYaw(-np.pi / 2, 0, 0),
                [rng.uniform(-0.25, 0.15), rng.uniform(-0.6, -0.4), 0.3],
            )
        # else:
        #     # Place in X bin:
        #     X_G["place"] = RigidTransform(
        #         RollPitchYaw(-np.pi / 2, 0, np.pi / 2),
        #         [rng.uniform(0.35, 0.65), rng.uniform(-0.12, 0.28), 0.3],
        #     )
        
        print(X_G)
        #print(mode)

        X_G, times = MakeGripperFrames(X_G, t0=context.get_time())
        print(
            f"Planned {times['postplace'] - times['initial']} second trajectory in mode {mode} at time {context.get_time()}."
        )
        state.get_mutable_abstract_state(int(self._times_index)).set_value(times)

        if False:  # Useful for debugging
            AddMeshcatTriad(meshcat, "X_Oinitial", X_PT=X_O["initial"])
            AddMeshcatTriad(meshcat, "X_Gprepick", X_PT=X_G["prepick"])
            AddMeshcatTriad(meshcat, "X_Gpick", X_PT=X_G["pick"])
            AddMeshcatTriad(meshcat, "X_Gplace", X_PT=X_G["place"])

        traj_X_G = MakeGripperPoseTrajectory(X_G, times)
        traj_wsg_command = MakeGripperCommandTrajectory(times)

        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(traj_X_G)
        state.get_mutable_abstract_state(int(self._traj_wsg_index)).set_value(
            traj_wsg_command
        )

    def start_time(self, context):
        return (
            context.get_abstract_state(int(self._traj_X_G_index))
            .get_value()
            .start_time()
        )

    def end_time(self, context):
        return (
            context.get_abstract_state(int(self._traj_X_G_index)).get_value().end_time()
        )

    def CalcGripperPose(self, context, output):
        context.get_abstract_state(int(self._mode_index)).get_value()

        traj_X_G = context.get_abstract_state(int(self._traj_X_G_index)).get_value()
        if traj_X_G.get_number_of_segments() > 0 and traj_X_G.is_time_in_range(
            context.get_time()
        ):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.set_value(
                context.get_abstract_state(int(self._traj_X_G_index))
                .get_value()
                .GetPose(context.get_time())
            )
            return

        # Command the current position (note: this is not particularly good if the velocity is non-zero)
        output.set_value(
            self.get_input_port(0).Eval(context)[int(self._gripper_body_index)]
        )

    def CalcWsgPosition(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        opened = np.array([0.107])
        np.array([0.0])

        if mode == PlannerState.GO_HOME:
            # Command the open position
            output.SetFromVector([opened])
            return

        traj_wsg = context.get_abstract_state(int(self._traj_wsg_index)).get_value()
        if traj_wsg.get_number_of_segments() > 0 and traj_wsg.is_time_in_range(
            context.get_time()
        ):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.SetFromVector(traj_wsg.value(context.get_time()))
            return

        # Command the open position
        output.SetFromVector([opened])

    def CalcControlMode(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(InputPortIndex(2))  # Go Home
        else:
            output.set_value(InputPortIndex(1))  # Diff IK

    def CalcDiffIKReset(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(True)
        else:
            output.set_value(False)

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index),
            self.get_input_port(int(self._iiwa_position_index)).Eval(context),
        )

    def CalcIiwaPosition(self, context, output):
        traj_q = context.get_mutable_abstract_state(int(self._traj_q_index)).get_value()

        output.SetFromVector(traj_q.value(context.get_time()))


from pydrake.all import (
    Fields, BaseField, DepthRenderCamera, RenderCameraCore, CameraInfo, ClippingRange, DepthRange
)
import torchvision
from torchvision.transforms import functional as Tf
import torch

from copy import deepcopy

from pydrake.systems.sensors import ImageRgba8U, ImageDepth32F
import matplotlib.pyplot as plt

class ImageProcessorSystem(LeafSystem):
    def __init__(self, idx, cam_info, model, device, meshcat, builder):
        super().__init__()
        self.idx = idx
        self.model = model
        self.device = device
        self.cam_info = cam_info
        self.meshcat = meshcat
        self.builder = builder

        # Declare input ports for the RGB images, depth images, and body poses from the camera
        self.DeclareAbstractInputPort("color_image", AbstractValue.Make(ImageRgba8U(480, 640)))
        self.DeclareAbstractInputPort("depth_image", AbstractValue.Make(ImageDepth32F(480, 640)))
        self.DeclareAbstractInputPort("body_pose_in_world", AbstractValue.Make(RigidTransform()))

        # Declare output port for the processed point cloud
        self.DeclareAbstractOutputPort("point_cloud_filtered", lambda: AbstractValue.Make(PointCloud(0, Fields(BaseField.kXYZs | BaseField.kRGBs))), self.CalcPointCloudOutput)

    def project_depth_to_pC(self, depth_pixel):
        v = depth_pixel[:, 0]
        u = depth_pixel[:, 1]
        Z = depth_pixel[:, 2]
        cx = self.cam_info.center_x()
        cy = self.cam_info.center_y()
        fx = self.cam_info.focal_x()
        fy = self.cam_info.focal_y()
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        pC = np.c_[X, Y, Z]
        return pC

    def process_point_cloud(self, rgb_image, depth_image, X_WC):
        
        # ycb = [
        #     "003_cracker_box.sdf",
        #     "004_sugar_box.sdf",
        #     "005_tomato_soup_can.sdf",
        #     "006_mustard_bottle.sdf",
        #     "009_gelatin_box.sdf",
        #     "010_potted_meat_can.sdf",
        # ]

        idx_to_item = {0:"CRAKER BOX", 1:"SUGAR BOX", 2:"TOMATO SOUP CAN", 3:"YELLOW MUSTARD BOTTLE",
                       4:"GELATIN BOX", 5:"POTTED MEAT CAN (SPAM)"}

        # change this for the robot to pick something different - 0 to 5 as above
        desired_object_ycb_idx = 5


        if running_as_notebook:
            with torch.no_grad():
                prediction = self.model([Tf.to_tensor(rgb_image[:, :, :3]).to(self.device)])
                for k in prediction[0].keys():
                    if k == "masks":
                        prediction[0][k] = (
                            prediction[0][k].mul(255).byte().cpu().numpy()
                        )
                    else:
                        prediction[0][k] = prediction[0][k].cpu().numpy()
        print(self.idx, prediction[0]["labels"])

        # plt.imshow(mask)
        # plt.title("Mask from Camera " + str(self.idx))
        # plt.colorbar()
        # plt.show()
        
        # with torch.no_grad():
        #     prediction = self.model([Tf.to_tensor(rgb_image[:, :, :3]).to(self.device)])[0]  # Exclude alpha channel
        
        if desired_object_ycb_idx not in prediction[0]['labels']:
            assert np.isinf(
                0.0
            ), f"No more {idx_to_item[desired_object_ycb_idx]} found. Deliberate Termination"

        mask_idx = np.argmax(prediction[0]['labels'] == desired_object_ycb_idx)  # Assuming 3 is the object label
        mask = prediction[0]['masks'][mask_idx, 0]

        mask_pixels = np.where(mask > 150)
        depth_values = depth_image[mask_pixels[0], mask_pixels[1]]
        depth_pixels = np.c_[mask_pixels[0], mask_pixels[1], depth_values]
        spatial_points = self.project_depth_to_pC(depth_pixels).T

        rgb_values = rgb_image[mask_pixels[0], mask_pixels[1], :3]
        rgb_points = rgb_values.T

        spatial_points = X_WC.multiply(spatial_points)

        N = spatial_points.shape[1]
        pcd = PointCloud(N, Fields(BaseField.kXYZs | BaseField.kRGBs))
        pcd.mutable_xyzs()[:] = spatial_points
        pcd.mutable_rgbs()[:] = rgb_points
        pcd.EstimateNormals(radius=0.1, num_closest=30)
        pcd.FlipNormalsTowardPoint(X_WC.translation())

        return pcd

    def CalcPointCloudOutput(self, context, output):
        # no data or squeeze?
        rgb_image = self.GetInputPort("color_image").Eval(context).data
        depth_image = deepcopy(self.GetInputPort("depth_image").Eval(context).data.squeeze())
        X_WC = self.GetInputPort("body_pose_in_world").Eval(context)

        point_cloud = self.process_point_cloud(rgb_image, depth_image, X_WC)
        output.set_value(point_cloud)


from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def CustomAddPointClouds(
    scenario,
    station,
    model,
    device,
    builder: DiagramBuilder,
    poses_output_port= None,
    meshcat = None,
):
    """
    Adds one DepthImageToPointCloud system to the `builder` for each camera in `scenario`, and connects it to the respective camera station output ports.

    Args:
        scenario: A Scenario structure, populated using the `LoadScenario` method.

        station: A HardwareStation system (e.g. from MakeHardwareStation) that has already been added to `builder`.

        builder: The DiagramBuilder containing `station` into which the new systems will be added.

        poses_output_port: (optional) HardwareStation will have a body_poses output port iff it was created with `hardware=False`. Alternatively, one could create a MultibodyPositionsToGeometryPoses system to consume the position measurements; this optional input can be used to support that workflow.

        meshcat: If not None, then a MeshcatPointCloudVisualizer will be added to the builder using this meshcat instance.

    Returns:
        A mapping from camera name to the DepthImageToPointCloud system.
    """

    # to_point_cloud = dict()
    to_point_cloud_filtered = dict()
    idx = 0
    for _, config in scenario.cameras.items():
        if not config.depth:
            return

        plant = station.GetSubsystemByName("plant")
        # frame names in local variables:
        # P for parent frame, B for base frame, C for camera frame.

        # Extract the camera extrinsics from the config struct.
        P = (
            GetScopedFrameByName(plant, config.X_PB.base_frame)
            if config.X_PB.base_frame
            else plant.world_frame()
        )
        X_PC = config.X_PB.GetDeterministicValue()

        # convert mbp frame to geometry frame
        body = P.body()
        plant.GetBodyFrameIdIfExists(body.index())
        # assert body_frame_id.has_value()

        X_BP = P.GetFixedPoseInBodyFrame()
        X_BC = X_BP @ X_PC

        intrinsics = CameraInfo(
            config.width,
            config.height,
            config.focal_x(),
            config.focal_y(),
            config.principal_point()[0],
            config.principal_point()[1],
        )

        # to_point_cloud[config.name] = builder.AddSystem(
        #     DepthImageToPointCloud(
        #         camera_info=intrinsics,
        #         fields=BaseField.kXYZs | BaseField.kRGBs,
        #     )
        # )
        # to_point_cloud[config.name].set_name(f"{config.name}.point_cloud")

        image_processor_system = ImageProcessorSystem(idx, intrinsics, model, device, meshcat, builder)
        idx = idx + 1
        to_point_cloud_filtered[config.name] = builder.AddSystem(image_processor_system)
        to_point_cloud_filtered[config.name].set_name(f"{config.name}.point_cloud_filtered")

        # builder.Connect(
        #     station.GetOutputPort(f"{config.name}.depth_image"),
        #     to_point_cloud[config.name].depth_image_input_port(),
        # )
        builder.Connect(
            station.GetOutputPort(f"{config.name}.depth_image"),
            to_point_cloud_filtered[config.name].GetInputPort("depth_image"),
        )
        # builder.Connect(
        #     station.GetOutputPort(f"{config.name}.rgb_image"),
        #     to_point_cloud[config.name].color_image_input_port(),
        # )
        builder.Connect(
            station.GetOutputPort(f"{config.name}.rgb_image"),
            to_point_cloud_filtered[config.name].GetInputPort("color_image"),
        )

        if poses_output_port is None:
            # Note: this is a cheat port; it will only work in single process
            # mode.
            poses_output_port = station.GetOutputPort("body_poses")

        camera_pose = builder.AddSystem(ExtractPose(int(body.index()), X_BC))
        camera_pose.set_name(f"{config.name}.pose")
        builder.Connect(
            poses_output_port,
            camera_pose.get_input_port(),
        )
        # builder.Connect(
        #     camera_pose.get_output_port(),
        #     to_point_cloud[config.name].GetInputPort("camera_pose"),
        # )
        builder.Connect(
            camera_pose.get_output_port(),
            to_point_cloud_filtered[config.name].GetInputPort("body_pose_in_world"),
        )
    # to_point_cloud also was return before
    return to_point_cloud_filtered



def clutter_clearing_demo():

    # add the NN model to pass to CustomAddPointClouds and image processors
    if running_as_notebook:

        def get_instance_segmentation_model(num_classes):
            # load an instance segmentation model pre-trained on COCO
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, progress=False
            )

            # get the number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            # now get the number of input features for the mask classifier
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            # and replace the mask predictor with a new one
            model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask, hidden_layer, num_classes
            )

            return model

    num_classes = 7
    model = get_instance_segmentation_model(num_classes)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load("models/clutter_maskrcnn_model.pt", map_location=device))
    model.eval()

    model.to(device)


    meshcat.Delete()
    builder = DiagramBuilder()

    scenario = LoadScenario(
        filename=FindResource("models/clutter.scenarios.yaml"),
        scenario_name="Clutter",
    )

    # change this to change the number of objects in the bin
    num_of_objs = 7

    model_directives = """
directives:
"""
    for i in range(num_of_objs if running_as_notebook else 10):
        object_num = rng.integers(0, len(ycb))
        # or "mustard_bottle" in ycb[object_num]
        if "cracker_box" in ycb[object_num]:
            # skip it. it's just too big! or too curvy
            continue
        model_directives += f"""
- add_model:
    name: ycb{i}
    file: package://manipulation/hydro/{ycb[object_num]}
"""
    scenario = AppendDirectives(scenario, data=model_directives)

    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    to_point_cloud_filtered = CustomAddPointClouds(scenario=scenario, station=station, meshcat = meshcat,
                                                                    builder=builder, model = model, device = device)
    plant = station.GetSubsystemByName("plant")

    # y_bin_grasp_selector = builder.AddSystem(
    #     GraspSelector(
    #         plant,
    #         plant.GetModelInstanceByName("bin0"),
    #         camera_body_indices=[
    #             plant.GetBodyIndices(plant.GetModelInstanceByName("camera0"))[0],
    #             plant.GetBodyIndices(plant.GetModelInstanceByName("camera1"))[0],
    #             plant.GetBodyIndices(plant.GetModelInstanceByName("camera2"))[0],
    #         ],
    #     )
    # )
    # builder.Connect(
    #     to_point_cloud["camera0"].get_output_port(),
    #     y_bin_grasp_selector.get_input_port(0),
    # )
    # builder.Connect(
    #     to_point_cloud["camera1"].get_output_port(),
    #     y_bin_grasp_selector.get_input_port(1),
    # )
    # builder.Connect(
    #     to_point_cloud["camera2"].get_output_port(),
    #     y_bin_grasp_selector.get_input_port(2),
    # )
    # builder.Connect(
    #     station.GetOutputPort("body_poses"),
    #     y_bin_grasp_selector.GetInputPort("body_poses"),
    # )

    x_bin_grasp_selector = builder.AddSystem(
        GraspSelector(
            plant,
            plant.GetModelInstanceByName("bin1"),
            camera_body_indices=[
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera3"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera4"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera5"))[0],
            ],
        )
    )
    builder.Connect(
        to_point_cloud_filtered["camera3"].get_output_port(),
        x_bin_grasp_selector.get_input_port(0),
    )
    builder.Connect(
        to_point_cloud_filtered["camera4"].get_output_port(),
        x_bin_grasp_selector.get_input_port(1),
    )
    builder.Connect(
        to_point_cloud_filtered["camera5"].get_output_port(),
        x_bin_grasp_selector.get_input_port(2),
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        x_bin_grasp_selector.GetInputPort("body_poses"),
    )

    # visualize filtered point cloud from cam 3, 4, 5

    # point_cloud_visualizer_cloud3 = builder.AddSystem(
    #     MeshcatPointCloudVisualizer(meshcat, f"camera3.cloud_filtered")
    # )
    # builder.Connect(
    #     to_point_cloud_filtered["camera3"].GetOutputPort("point_cloud_filtered"),
    #     point_cloud_visualizer_cloud3.cloud_input_port(),
    # )

    # point_cloud_visualizer_cloud4 = builder.AddSystem(
    #     MeshcatPointCloudVisualizer(meshcat, f"camera4.cloud_filtered")
    # )
    # builder.Connect(
    #     to_point_cloud_filtered["camera4"].GetOutputPort("point_cloud_filtered"),
    #     point_cloud_visualizer_cloud4.cloud_input_port(),
    # )

    # point_cloud_visualizer_cloud5 = builder.AddSystem(
    #     MeshcatPointCloudVisualizer(meshcat, f"camera5.cloud_filtered")
    # )
    # builder.Connect(
    #     to_point_cloud_filtered["camera5"].GetOutputPort("point_cloud_filtered"),
    #     point_cloud_visualizer_cloud5.cloud_input_port(),
    # )

    planner = builder.AddSystem(Planner(plant))
    builder.Connect(
        station.GetOutputPort("body_poses"), planner.GetInputPort("body_poses")
    )
    builder.Connect(
        x_bin_grasp_selector.get_output_port(),
        planner.GetInputPort("x_bin_grasp"),
    )
    # builder.Connect(
    #     y_bin_grasp_selector.get_output_port(),
    #     planner.GetInputPort("y_bin_grasp"),
    # )
    builder.Connect(
        station.GetOutputPort("wsg.state_measured"),
        planner.GetInputPort("wsg_state"),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        planner.GetInputPort("iiwa_position"),
    )

    robot = station.GetSubsystemByName("iiwa_controller_plant_pointer_system").get()

    # Set up differential inverse kinematics.
    diff_ik = AddIiwaDifferentialIK(builder, robot)
    builder.Connect(planner.GetOutputPort("X_WG"), diff_ik.get_input_port(0))
    builder.Connect(
        station.GetOutputPort("iiwa.state_estimated"),
        diff_ik.GetInputPort("robot_state"),
    )
    builder.Connect(
        planner.GetOutputPort("reset_diff_ik"),
        diff_ik.GetInputPort("use_robot_state"),
    )

    builder.Connect(
        planner.GetOutputPort("wsg_position"),
        station.GetInputPort("wsg.position"),
    )

    # The DiffIK and the direct position-control modes go through a PortSwitch
    switch = builder.AddSystem(PortSwitch(7))
    builder.Connect(diff_ik.get_output_port(), switch.DeclareInputPort("diff_ik"))
    builder.Connect(
        planner.GetOutputPort("iiwa_position_command"),
        switch.DeclareInputPort("position"),
    )
    builder.Connect(switch.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(
        planner.GetOutputPort("control_mode"),
        switch.get_port_selector_input_port(),
    )

    builder.AddSystem(StopButton(meshcat))

    diagram = builder.Build()

    simulator = Simulator(diagram)
    context = simulator.get_context()

    plant_context = plant.GetMyMutableContextFromRoot(context)
    z = 0.2
    for body_index in plant.GetFloatingBaseBodies():
        tf = RigidTransform( 
            UniformlyRandomRotationMatrix(generator),
            [rng.uniform(0.35, 0.65), rng.uniform(-0.12, 0.28), z],
        )
        plant.SetFreeBodyPose(plant_context, plant.get_body(body_index), tf)
        z += 0.1

    simulator.AdvanceTo(np.inf)
    meshcat.Flush()  # Wait for the large object meshes to get to meshcat.

    if running_as_notebook:
        simulator.set_target_realtime_rate(1.0)
        simulator.AdvanceTo(np.inf)


clutter_clearing_demo()