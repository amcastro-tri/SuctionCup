"""Simple suction cup model"""

import argparse
from dataclasses import (dataclass, field)
from math import sqrt
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import DrakeVisualizer
from pydrake.geometry import ProximityProperties
from pydrake.geometry import (Box, Cylinder, HalfSpace, Sphere)
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.multibody.math import (
    SpatialForce,
    SpatialMomentum,
    SpatialVelocity,
    SpatialAcceleration,
)

# Multibody
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.plant import (
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    ApplyMultibodyPlantConfig,
    CoulombFriction,
    DiscreteContactSolver,
    MultibodyPlantConfig,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.tree import (
    PrismaticJoint,
    RotationalInertia,
    SpatialInertia,
    UnitInertia,
    RigidBody,    
)

from pydrake.multibody.tree import (ModelInstanceIndex, BodyIndex)
from typing import List

# Systems
from pydrake.systems.framework import DiagramBuilder, EventStatus
from pydrake.systems.analysis import (PrintSimulatorStatistics, Simulator)
from pydrake.systems.primitives import (
    MatrixGain,    
    ConstantVectorSource,
    Multiplexer)


# Misc.
from pydrake.all import (MeshcatVisualizer, StartMeshcat, MeshcatVisualizerParams)
from pydrake.all import (AbstractValue,
                         LeafSystem,
                         Rgba)

def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)

@dataclass
class ContactProperties:
    stiffness: float
    hc_dissipation: float
    relaxation_time: float
    friction: float

@dataclass
class BoxParams:
    name : str = "box"
    size : float = 0.3
    mass : float = 2.0
    contact_properties = ContactProperties(1.0e6, 10.0, 1.0e-4, 0.3)

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
    plant.RegisterCollisionGeometry(
      plant.world_body(), RigidTransform.Identity(),
      HalfSpace(), "ground_collision", properties)
    plant.RegisterVisualGeometry(
      plant.world_body(), RigidTransform([0, 0, -0.05]),
      Box(5, 5, 0.1), "ground_visual", [0, 0, 1, 0.3])


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
        
    green = [0,1,0,1]    
    shape = Box(size, size, size)
    plant.RegisterCollisionGeometry(
      body, RigidTransform.Identity(),
      shape, name + "_collision", properties)
    plant.RegisterVisualGeometry(
      body, RigidTransform.Identity(),
      shape, name + "_visual", green)    
    return body

def InitializeBox(box, params, plant, plant_context):
    theta = 20 # Initial pitch angle for the box 
    x = np.sin(theta/180.0*np.pi)*params.size/2.0
    z = params.size
    X_WB = xyz_rpy_deg([x, 0.0, z], [0.0, theta, 0.0])
    plant.SetFreeBodyPose(
        plant_context, box, X_WB)
    V_WB = SpatialVelocity(w=[0, 0, 0], v=[0.0, 0, 0]) 
    plant.SetFreeBodySpatialVelocity(box, V_WB, plant_context)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target_realtime_rate", type=float, default=0.0,
        help="Desired rate relative to real time.  See documentation for "
             "Simulator::set_target_realtime_rate() for details.")
    parser.add_argument(
        "--simulation_time", type=float, default=1.0,
        help="Desired duration of the simulation in seconds.")
    parser.add_argument(
        "--time_step", type=float, default=5.0e-3,
        help="If greater than zero, the plant is modeled as a system with "
             "discrete updates and period equal to this time_step. "
             "If 0, the plant is modeled as a continuous system.")
    parser.add_argument(
        "--meshcat", type=bool, default=False,
        help="Adds Meshcat viz.")
    parser.add_argument("--contact_model",
                        type=str,
                        default="tamsi",
                        help="Discrete contact model, one of [tamsi, sap, convex, lagged]")
    args = parser.parse_args()
    
    # Run sim for the given set of command line args.
    SuctionCupSim(args)    

# Define the suction cup system
class SuctionCupSystem(LeafSystem):
    def __init__(self, body, X_BCup, radius, contact_properties):
        LeafSystem.__init__(self)
        self.body = body
        self.X_BCup = X_BCup
        self.radius = radius
        self.contact_properties = contact_properties

        state_index = self.DeclareContinuousState(1)  # One state variable.
        self.DeclareStateOutputPort("y", state_index)  # One output: y=x.

    # xdot(t) = -x(t) + x^3(t)
    def DoCalcTimeDerivatives(self, context, derivatives):
        x = context.get_continuous_state_vector().GetAtIndex(0)
        xdot = -x + x**3
        derivatives.get_mutable_vector().SetAtIndex(0, xdot)

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
        stiffness=1.0e16, hc_dissipation=0.0,
        relaxation_time=0.0, friction=0.5)
    AddGround(ground_contact_properties, plant)
    
    box_params = BoxParams("cylinder")
    box = AddBox(box_params, plant)

    plant.Finalize()

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

    InitializeBox(box, box_params, plant, plant_context)    
    
    file_name = "suction_cup_{}_{}".format(args.contact_model,args.time_step).replace(".","p") + ".dat"
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
    simulator.AdvanceTo(args.simulation_time)
    if args.meshcat:
        meshcat.StopRecording()
        meshcat.PublishRecording()
    f.close()
    #PrintSimulatorStatistics(simulator)


if __name__ == "__main__":
    main()
