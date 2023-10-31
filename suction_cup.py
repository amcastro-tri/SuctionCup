"""Ad-hoc suction cup model"""

import argparse
from dataclasses import (dataclass, field)
from math import sqrt
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from pydrake.common.cpp_param import List
from pydrake.common import FindResourceOrThrow
from pydrake.common.value import Value
from pydrake.geometry import (
    AddCompliantHydroelasticProperties,
    AddRigidHydroelasticProperties, 
    AddCompliantHydroelasticPropertiesForHalfSpace, 
    Box, 
    Cylinder,
    DrakeVisualizer, 
    HalfSpace, 
    ProximityProperties, 
    QueryObject, 
    Sphere)
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.multibody.math import (
    SpatialForce,
    SpatialMomentum,
    SpatialVelocity,
    SpatialAcceleration,
)
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.plant import (
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    ApplyMultibodyPlantConfig,
    CoulombFriction,
    DiscreteContactSolver,
    ExternallyAppliedSpatialForce,
    MultibodyPlantConfig,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.tree import (
    BodyIndex,
    ModelInstanceIndex,
    PrismaticJoint,
    RotationalInertia,
    SpatialInertia,
    UnitInertia,
    RigidBody,    
)
from pydrake.systems.framework import DiagramBuilder, EventStatus
from pydrake.systems.analysis import (PrintSimulatorStatistics, Simulator)
from pydrake.systems.primitives import (
    MatrixGain,    
    ConstantVectorSource,
    Multiplexer,
    Sine)
from pydrake.all import (MeshcatVisualizer, StartMeshcat, MeshcatVisualizerParams)
from pydrake.all import (AbstractValue,
                         LeafSystem,
                         Rgba)

def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)

# Struct to store contact properties.
@dataclass
class ContactProperties:
    hydro_modulus: float
    stiffness: float
    hc_dissipation: float
    relaxation_time: float
    friction: float

# Struct to store the box's model parameters.
@dataclass
class BoxParams:
    name : str = "box"
    size : float = 0.3
    mass : float = 2.0
    contact_properties = ContactProperties(5.0e4, 1.0e6, 10.0, 2.0e-1, 0.3)

# Adds a model of the ground to `plant` with `contact_properties`, of type
# ContactProperties.
def AddGround(contact_properties, plant):
    properties = ProximityProperties()
    properties.AddProperty(
        "material", "point_contact_stiffness", contact_properties.stiffness)
    properties.AddProperty(
        "material", "hunt_crossley_dissipation", contact_properties.hc_dissipation)        
    properties.AddProperty(
        "material", "relaxation_time", contact_properties.relaxation_time)
    properties.AddProperty(
        "material", "coulomb_friction",
        CoulombFriction(
            contact_properties.friction,
            contact_properties.friction))    
    
    AddCompliantHydroelasticPropertiesForHalfSpace(
            slab_thickness=0.1, hydroelastic_modulus=contact_properties.hydro_modulus,
            properties=properties)

    plant.RegisterCollisionGeometry(
      plant.world_body(), RigidTransform.Identity(),
      HalfSpace(), "ground_collision", properties)
    plant.RegisterVisualGeometry(
      plant.world_body(), RigidTransform([0, 0, -0.05]),
      Box(5, 5, 0.1), "ground_visual", [0, 0, 1, 0.3])

# Adds a box model to `plant` given `params`, of type BoxParams.
def AddBox(params, plant):
    name = params.name
    size = params.size
    mass = params.mass
    I_Bo = mass * UnitInertia.SolidBox(size, size, size)
    M_Bo = SpatialInertia.MakeFromCentralInertia(
        mass, np.array([0, 0, 0]), I_Bo)
    body = plant.AddRigidBody(name, M_Bo)

    # Contact properties.
    contact_properties = params.contact_properties
    properties = ProximityProperties()
    properties.AddProperty(
        "material", "point_contact_stiffness", contact_properties.stiffness)
    properties.AddProperty(
        "material", "hunt_crossley_dissipation", contact_properties.hc_dissipation)    
    properties.AddProperty(
        "material", "relaxation_time", contact_properties.relaxation_time)
    properties.AddProperty(
        "material", "coulomb_friction",
        CoulombFriction(
            contact_properties.friction,
            contact_properties.friction))
    
    AddRigidHydroelasticProperties(resolution_hint=0.01, properties=properties)
        
    green = [0,1,0,1]    
    shape = Box(size, size, size)
    plant.RegisterCollisionGeometry(
      body, RigidTransform.Identity(),
      shape, name + "_collision", properties)
    plant.RegisterVisualGeometry(
      body, RigidTransform.Identity(),
      shape, name + "_visual", green)    
    return body

# Initializes the pose of the box.
def InitializeBox(box, params, plant, plant_context):
    theta = 20 # Initial pitch angle for the box 
    x = np.sin(theta/180.0*np.pi)*params.size/2.0
    z = 0.3
    X_WB = xyz_rpy_deg([x, 0.0, z], [0.0, theta, 0.0])
    plant.SetFreeBodyPose(
        plant_context, box, X_WB)
    V_WB = SpatialVelocity(w=[0, 0, 0], v=[0.0, 0, 0]) 
    plant.SetFreeBodySpatialVelocity(box, V_WB, plant_context)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target_realtime_rate", type=float, default=1.0,
        help="Desired rate relative to real time.  See documentation for "
             "Simulator::set_target_realtime_rate() for details.")
    parser.add_argument(
        "--simulation_time", type=float, default=10.0,
        help="Desired duration of the simulation in seconds.")
    parser.add_argument(
        "--time_step", type=float, default=0.005,
        help="If greater than zero, the plant is modeled as a system with "
             "discrete updates and period equal to this time_step. "
             "If 0, the plant is modeled as a continuous system.")
    parser.add_argument(
        "--meshcat", type=bool, default=True,
        help="Adds Meshcat viz.")
    args = parser.parse_args()
    
    # Run sim for the given set of command line args.
    SuctionCupSim(args)    

# Struct to store the parameters for a SuctionCupSystem model.
@dataclass
class SuctionCupParams:
    # Pose of the cup's frame C in the body B on which it attaches.
    # The cup's frame z-axes defines the "cup's normal", which points towards
    # the outside of the cup, opposite of the airflow direction. This allows the
    # model to consider the "direction" of the flow.
    X_BCup = RigidTransform()

    # The cup's geometry is simplified to a small sphere with this radius. Once
    # objects make contact with this sphere, the suction force will reach it's
    # maximum f_max (see below).
    radius : float = 0.01

    # Maximum contact force. In this ad-hoc model, for is inversely proportional
    # to the distance to the cup, with a maximum force f_max when an external
    # object makes contact with the cup (idealized to a single sphere).
    f_max : float = 200.0

    # Models a damping torque proportional to the relative angular velocity
    # between the cup body and the body being manipulated.
    damping : float = 0.2

    # Contact properties of the idealized geometry of the cup, a sphere.
    # Compliance of the cup can be modeled by adjusting these compliant
    # parameters.
    contact_properties = ContactProperties(
        hydro_modulus=5.0e4, 
        stiffness=1.0e5, hc_dissipation=10.0, relaxation_time=1.0e-4,
        friction=0.3)

# System used to model a single suction cup attached on a rigid body.
#
# The cup is idealized to a single point. The "direction" in which the cup
# opening is facing is defined by its normal. The normal in turn is defined to
# be the z-axes of the cup's frame, defined at construction given its pose in
# the cup's body (X_BCup). With these definitions, this system implements an
# ad-hoc model that includes:
#  - Action inversely proportional to the distance,
#  - Maximum force is f_max, modeling the situation in which the cup is fully
#    blocked by the object (f_max in reality is given by the suction pressure
#    times the cup's area)
#  - Directionality is considered. That is, objects "behind" the cup experience
#    no force. Objects in front, will experience a force proportional to the
#    cosine of the angle between the cup's normal and the position vector
#    from the cup's center C to the nearest point on the object N.
#  - We include a damping term, modeling dissipation by deformations within the
#    elastic body of the cup. This term is modeled as proportional to the
#    relative angular velocity between body B and manipuland body A.
#
# We can write this ad-hoc model mathematically as:
#   p_hat = p_NC/distance
#   cos(theta) = dot(p_hat, n_hat)
#   f_AN =  p_hat * min(1.0, radius/distance) * cos(theta)
#   t_A = -damping * w_BA * Heaviside(2 * radius - distance)
#
# where p_NC is the position from the cup's center to the nearest point on a
# manipuland, distance = p_NC.norm(), n_hat is the cup's normal (facing towards
# the outside, opposite to the air flow direction), f_BN is the force on
# manipuland body A applied at the nearest point N and t_A is is the damping
# torque to model dissipation.
#
# All these terms are ad-hoc for demonstration purposes only. Each of these
# terms can be improved using fluid mechanics models and/or experimental data.
class SuctionCupSystem(LeafSystem):
    def __init__(self, plant, body, cup_params):
        LeafSystem.__init__(self)
        self.plant = plant
        self.body = body
        self.X_BCup = cup_params.X_BCup
        self.radius = cup_params.radius
        self.f_max = cup_params.f_max
        self.damping = cup_params.damping
        self.contact_properties = cup_params.contact_properties

        model_poses_list = Value[List[RigidTransform]]
        self.poses_input_port = self.DeclareAbstractInputPort(
            name="poses_input",
                model_value=model_poses_list())
        model_velocities_list = Value[List[SpatialVelocity]]
        self.velocities_input_port = self.DeclareAbstractInputPort(
            name="velocities_input",
                model_value=model_velocities_list())
        self.geometry_query_input_port = self.DeclareAbstractInputPort(
            name="geometry_query", model_value=Value(QueryObject()))
        model_forces = Value[List[ExternallyAppliedSpatialForce]]

        self.command_input_port = self.DeclareVectorInputPort("command", 1)

        self.spatial_forces_output_port = self.DeclareAbstractOutputPort("spatial_forces",
                                    lambda: model_forces(),
                                       self.CalcSuctionForces)

    def get_body_poses_input_port(self):
        return self.poses_input_port
    
    def get_body_velocities_input_port(self):
        return self.velocities_input_port
    
    def get_geometry_query_input_port(self):
        return self.geometry_query_input_port
    
    def get_spatial_forces_output_port(self):
        return self.spatial_forces_output_port
    
    def get_command_input_port(self):
        return self.command_input_port    
        
    # Makes a new cup system and defines the needed geometry in plant.
    # Note: This must be called pre-finalize.            
    def Make(builder, plant, body, cup_params):
        # Add contact geometry to model the cup. In this simple example, just a
        # sphere at the cup's location.
        contact_properties = cup_params.contact_properties
        properties = ProximityProperties()
        properties.AddProperty(
            "material", "point_contact_stiffness", contact_properties.stiffness)
        properties.AddProperty(
            "material", "hunt_crossley_dissipation", contact_properties.hc_dissipation)        
        properties.AddProperty(
            "material", "relaxation_time", contact_properties.relaxation_time)
        properties.AddProperty(
            "material", "coulomb_friction",
        CoulombFriction(
            contact_properties.friction,
            contact_properties.friction))
        shape = Sphere(cup_params.radius)
        plant.RegisterCollisionGeometry(
            body, cup_params.X_BCup,
            shape, "cup_collision", properties)
        plant.RegisterVisualGeometry(
            body, cup_params.X_BCup,
            shape, "cup_visual", [1, 0, 0, 1])
        
        return SuctionCupSystem(plant, body, cup_params)

    # Adds cup system to the diagram and connects it.
    # Note: This must be called post-finalize.
    def AddAndConnect(builder, plant, scene_graph, cup_system):        
        cup = builder.AddSystem(cup_system)
        builder.Connect(plant.get_body_poses_output_port(), cup.get_body_poses_input_port())
        builder.Connect(plant.get_body_spatial_velocities_output_port(), cup.get_body_velocities_input_port())
        builder.Connect(scene_graph.get_query_output_port(), cup.get_geometry_query_input_port())
        builder.Connect(cup.get_spatial_forces_output_port(), plant.get_applied_spatial_force_input_port())    

    def CalcSuctionForces(self, context, spatial_forces_vector):
            query_object = self.geometry_query_input_port.Eval(context)            
            poses = self.poses_input_port.Eval(context)
            velocities = self.velocities_input_port.Eval(context)
            command = self.get_command_input_port().Eval(context)[0]
            command_positive = max(0.0, command)
            #geometry_poses = self.plant.get_geometry_poses_output_port().Eval(context)
            X_BCup = self.X_BCup

            inspector = query_object.inspector()
            #f_id = inspector.GetFrameId(g_id)
            #body = self.plant.GetBodyFromFrameId(f_id)

            # Cup body B position and spatial velocity
            X_WB = poses[self.body.index()]
            p_WB = X_WB.translation()
            V_WB = velocities[self.body.index()]
            w_WB = V_WB.rotational()

            # Position of the cup C
            X_WCup = X_WB.multiply(X_BCup)
            p_WCup = X_WCup.translation()

            # We define the cup's normal as the z-axes of C.
            normal_W = X_WCup.rotation().matrix()[ :,2]

            all_sdfs = query_object.ComputeSignedDistanceToPoint(p_WCup)

            all_forces = List[ExternallyAppliedSpatialForce]()
            for sdf in all_sdfs:
                g_id = sdf.id_G
                p_GN = sdf.p_GN
                distance = sdf.distance

                # Geometry G attached to body A
                f_id = inspector.GetFrameId(g_id)
                bodyA = self.plant.GetBodyFromFrameId(f_id)
                X_AG = inspector.GetPoseInFrame(g_id)
                X_WA = poses[bodyA.index()]
                V_WA = velocities[bodyA.index()]
                w_WA = V_WA.rotational()
                p_WA = X_WA.translation()

                # Nearest point N on geometry G, in the world frame.
                X_WG = X_WA.multiply(X_AG)
                p_WN = X_WG.multiply(p_GN)

                # Relative velocity of A wrt B (to define dissipation)
                w_BA_W = w_WA - w_WB

                # Point from N to C
                p_NC_W = p_WCup - p_WN

                # Direction vector
                p_hat = p_NC_W / distance

                # Force on A at N
                F_AN_W = SpatialForce.Zero()                
                cos_theta = -np.dot(p_hat, normal_W)
                # Force is only positive if "in front" of the cup.
                f = 0.0
                if cos_theta > 0.0:
                    f = self.f_max * min(self.radius / distance, 1.0) * cos_theta
                    f_AN_W = f * p_hat * command_positive
                    # Dirty model of damping, only added if the box is in
                    # contact with the cup or "close enough".
                    t_A_W = np.array([0.0,0.0,0.0])
                    if distance < 2.0 * self.radius:
                        t_A_W = - self.damping * w_BA_W * command_positive
                    F_AN_W = SpatialForce(t_A_W, f_AN_W)

                # Position of nearest point N in A, expressed in World.
                R_AW = X_WA.rotation().inverse()
                p_AN_W =  p_WN - p_WA
                p_AN_A = R_AW.multiply(p_AN_W)

                force = ExternallyAppliedSpatialForce()
                force.body_index = bodyA.index()
                force.p_BoBq_B = p_AN_A
                force.F_Bq_W = F_AN_W
                
                all_forces.append(force)

            spatial_forces_vector.set_value(all_forces)            

def SuctionCupSim(args):    
    # Start the visualizer.
    if args.meshcat:
        meshcat = StartMeshcat()
        meshcat.Set2dRenderMode(xmin=-0.5, xmax=0.5, ymin=-0.1, ymax=1.0)

    # Build model.
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder=builder, time_step=args.time_step)
    config = MultibodyPlantConfig()
    config.time_step = args.time_step
    config.discrete_contact_solver = "sap"
    config.stiction_tolerance = 1.0e-4
    ApplyMultibodyPlantConfig(config, plant)

    ground_contact_properties = ContactProperties(
        hydro_modulus=1.0e6,
        stiffness=1.0e16, hc_dissipation=0.0,
        relaxation_time=0.0, friction=0.5)
    AddGround(ground_contact_properties, plant)
    
    box_params = BoxParams("box")
    box = AddBox(box_params, plant)    

    # In this simple example, the cup is anchored to the world body.
    cup_body = plant.world_body()
    # We define the cup's frame C with it's z-axes pointing down to define the
    # cup's normal (which defines the suction direction)    
    cup_params = SuctionCupParams()
    cup_params.X_BCup = xyz_rpy_deg([0.0,0.0,0.5], [0.0, 180.0, 0.0])
    cup_system = SuctionCupSystem.Make(builder, plant, cup_body, cup_params)
    plant.Finalize()
    SuctionCupSystem.AddAndConnect(builder, plant, scene_graph, cup_system)

    # Control amplitude of the suction force.
    #amplitude = 1.0
    #freq = 0.25 # in Hertz
    #omega = 2.0 * np.pi * freq    
    #forcing = builder.AddSystem(
    #    Sine(amplitudes=np.array([1.0]), frequencies=np.array([omega]), phases=np.array([0])))
    #builder.Connect(forcing.get_output_port(0),
    #                cup_system.get_command_input_port())


    # Add viz.
    if args.meshcat:
        viz_params = MeshcatVisualizerParams()
        viz_params.publish_period = args.time_step
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat, viz_params)

    # Done defining the diagram.
    diagram = builder.Build()

    # Create context and set initial condition.
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    cup_context = cup_system.GetMyMutableContextFromRoot(context)
    cup_system.get_command_input_port().FixValue(cup_context, [1.0])

    InitializeBox(box, box_params, plant, plant_context)    
    
    # Do some logging.
    file_name = "suction_cup_{}".format(args.time_step).replace(".","p") + ".dat"
    print(file_name)
    f = open(file_name, "w")
    def monitor(root_context):
        plant_context = plant.GetMyMutableContextFromRoot(root_context)
        X_WB = plant.EvalBodyPoseInWorld(plant_context, box)
        t = plant_context.get_time()
        x = X_WB.translation()[0]
        z = X_WB.translation()[2]        
        results = plant.get_contact_results_output_port().Eval(plant_context)
        nc = results.num_point_pair_contacts()
        fc = np.array([0, 0, 0])
        if nc == 1:
            pp_info = results.point_pair_contact_info(0)
            pp = pp_info.point_pair()
            vt = pp_info.slip_speed()
            fc = pp_info.contact_force()
        
        line = "{} {} {} {} {}\n".format(t, x, z, fc[0], fc[2])
        f.write(line)
        return EventStatus.Succeeded()    

    # Setup simulator
    simulator = Simulator(diagram, context)
    simulator.set_monitor(monitor=monitor)
    simulator.set_target_realtime_rate(args.target_realtime_rate)
    simulator.set_publish_every_time_step(True)
    simulator.Initialize()
    if args.meshcat:
        input("Press Enter to continue...")

    # Run sim.
    if args.meshcat:
        meshcat.StartRecording(frames_per_second=1.0/args.time_step)

    for t in np.arange(0.0, args.simulation_time, 0.1):
        if t > 5.0:
            cup_system.get_command_input_port().FixValue(cup_context, [0.0])
        simulator.AdvanceTo(t)

    if args.meshcat:
        meshcat.StopRecording()
        meshcat.PublishRecording()
    f.close()
    PrintSimulatorStatistics(simulator)


if __name__ == "__main__":
    main()
