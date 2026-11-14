# Wisp

A Computational Fluid Dynamics Simulation based on SPH, implemented using Vulkan + CUDA Interop

Perhaps, this is an extension to my ![2D Fluid Simulator](https://github.com/irrevocablesake/Ryutai). 

One fine Friday, I was hunting for a project idea to keep myself busy over the weekend. But, what was supposed to be a weekend project, turned into a 3 week fun journey of delving into Maths, Physics and Optimizations. All the effort was spent to implement a simulation model known as Smoothened Particle Hydro-dynamics ( Physics based ). And representing the model results as discrete particles. Eventually, a surface was approximated through fragment shader computations, known as Screen Space Fluid ( SSF ), to give the fluid a more uniform look.

The idea remains quite similar to our 2D simulation: Fluids flow. But this time, instead of representing the fluid using fields on a grid, we represent it using individual particles. Each particle represents a small portion of the fluid and carries properties such as position, velocity and mass. Together, these particles allow us to approximate the behavior of a continuous fluid.

At the highest level, the simulation revolves around three important physical properties:
- _**Density:**_ Tells us how much fluid mass is packed into a given region. It helps us understand how compressed or spread out the fluid is.
- _**Pressure:**_ Determines how particles push against their neighbours, resisting compression and helping maintain the fluid's volume.
- _**Viscosity:**_ Determines how much neighbouring particles resist relative motion, giving the fluid its characteristic resistance to flow.

A general flow of the program can be thought of as follows:
- _**Initialization:**_ We start with a lattice of particles, initially at rest. Each particle has an associated mass, and the initial arrangement establishes the fluid's resting density.
  - At every simulation _iteration_, we perform the following operations, updating the state of the particles based on their surroundings:
    - _**Compute density:**_ Each particle examines its neighbours to estimate the amount of fluid surrounding it.
    - _**Compute pressure:**_ Using the estimated density, we determine how much pressure each particle experiences.
    - _**Compute pressure forces:**_ We calculate the forces exerted by neighbouring particles due to pressure. These forces resist compression and drive the fluid apart where necessary.
    - _**Compute viscosity forces:**_ We calculate the resistance to relative motion between neighbouring particles, helping smooth out their velocities.
    - _**Update particles:**_ We combine the resulting forces with external forces such as gravity to update particle velocities and positions.

Additionally, we employ a few techniques to improve the behaviour of the simulation:
- _**XSPH:**_ Helps **smoothen** noisy particle velocities by blending a particle's velocity with those of its neighbours.
- _**Surface Tension:**_ Helps maintain **cohesion** between surface particles, preventing the fluid from looking overly scattered.
- _**Vortex Force:**_ Introduces **additional energy** into the system, helping sustain interesting rotational motion instead of allowing the fluid to quickly settle into equilibrium.

The result is a collection of particles whose collective behaviour _**approximates**_ a flowing 3D fluid.
