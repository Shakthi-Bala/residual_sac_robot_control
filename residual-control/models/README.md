# UR16e free-space model

`ur16e.xml` retains all six arm links, their inertias, joint transforms, and visual meshes from the UR16e model identified in `provenance.json`. The source was generated from the official Universal Robots description in the linked project. Mesh attribution and redistribution terms are in `assets/LICENSE`.

The extension removes the gripper/camera and adds a compact tool payload at `tool0`. Geometry has collisions disabled. Torque-limited actuators implement the simulated velocity servo; these are benchmark dynamics rather than a calibrated model of the UR controller. Payload variation changes mass and inertia in the physical MuJoCo model, while compensation uses a separately frozen nominal model. Command delay is an explicit FIFO at 50 Hz. Neither perturbation is an observation label available to the policies.

The benchmark uses a restricted joint envelope around one working posture. Mesh realism does not imply actuator fidelity, collision safety, or validation on a physical UR16e.
