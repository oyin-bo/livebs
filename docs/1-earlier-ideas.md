

## **Problem Statement**

The core problem is to simulate a massive number of particles and their interactions in real-time, leveraging the parallel processing power of the GPU. Specifically, we're focused on simulating gravitational and other attraction-repulsion forces. The goal is to create a performant and stable particle system that can handle a large number of particles and complex interactions.

## **Proposed Solutions**

We've explored three main approaches for handling the physics and data management:

---

### **1\. The Buffers and Transform Feedback Approach**

This was our initial, more conventional approach.

#### **Core Idea**

This method uses **WebGL2 buffers** to store particle data (positions, velocities, masses) and relies on **transform feedback** to update the data in a parallel fashion. The physics calculations are performed in a **vertex shader**, which reads from an input buffer and writes the results to an output buffer.

#### **Detailed Breakdown**

* **Data Storage:** Particle data is stored in gl.ARRAY\_BUFFERs. These buffers contain interleaved data, such as a particle's position (vec3) and velocity (vec3), and a separate buffer for static data like mass.  
* **Physics Calculation:** A vertex shader is used for the physics calculations. For each particle, a vertex is processed, and the shader reads the particle's current state from the input buffer.  
* **Transform Feedback:** This is the key mechanism. Instead of drawing to the screen, the results of the vertex shader are "fed back" and written directly to a designated output buffer.  
* **Ping-Pong Buffering:** We use two sets of buffers ("in" and "out"). After each physics step, the "out" buffer from the current frame becomes the "in" buffer for the next frame, effectively swapping them to enable continuous updates.

#### **Practical Caveats & Pitfalls**

* **Data Locality:** The interleaved data format can sometimes lead to poor memory access patterns, as the shader might not be reading all the data it needs in a contiguous block.  
* **Neighbor Finding:** This approach is not ideal for finding neighbors efficiently, as gl\_VertexID provides a linear index, not a spatial one. This leads to complex or inefficient brute-force neighbor search algorithms, as seen in the initial gl\_physics.js code with its for loops.  
* **Transform Feedback Limitations:** While powerful, transform feedback is primarily designed for vertex processing. Trying to use it for complex, non-vertex-related tasks can be clunky.

---

### **2\. The Textures and Render-to-Texture Approach**

This is the approach we're currently pursuing.

#### **Core Idea**

This method shifts the data storage from buffers to **WebGL2 textures**. The physics calculations are performed in a **fragment shader**, and the updated data is written to a set of output textures using **Multiple Render Targets (MRT)**.

#### **Detailed Breakdown**

* **Data Storage:** We use separate 2D textures for each particle attribute: positionsTexture (RGBA32F), velocitiesTexture (RGBA32F), massesTexture (R32F), and fishTexture (R32I). The 2D nature allows for a larger number of particles and more flexible data layout than a 1D buffer.  
* **Physics Calculation:** A fragment shader is the workhorse here. We render a single quad that covers the entire viewport, and for each pixel (or fragment), the shader calculates the physics for a corresponding particle.  
* **MRT and Render-to-Texture:** Instead of transform feedback, we use a framebuffer with multiple color attachments. The fragment shader writes the updated position to one attachment, velocity to another, and so on.  
* **Ping-Pong Textures:** Similar to the buffer approach, we use "in" and "out" texture sets to enable a continuous simulation loop.

#### **Practical Caveats & Pitfalls**

* **Texture Size Limits:** We must be mindful of the maximum texture size supported by the hardware, which can limit the total number of particles. We've used the gl.getParameter(gl.MAX\_TEXTURE\_SIZE) to determine a safe stride.  
* **Fragment Shader Complexity:** Fragment shaders are excellent for parallel tasks, but complex logic can be difficult to manage.  
* **Integer Texture Support:** We're using gl.R32I for the fishTexture. While this is a WebGL2 feature, support can vary slightly, and we need to ensure the RED\_INTEGER format is properly handled.

---

### **3\. The "Precut" Approach (The Menace)**

This approach was a theoretical exploration of a highly unorthodox but potentially powerful method.

#### **Core Idea**

This method uses a complex combination of **dynamically generated shaders**, **quadtree-based data structures stored in textures**, and a series of **multi-pass rendering steps** to optimize many-body force calculations.

#### **Detailed Breakdown**

* **Quadtree in Textures:** The core innovation is representing a quadtree entirely within a set of textures. Each level of the quadtree gets its own texture, and the pixels in these textures store aggregated data (e.g., center of mass, total mass) for that level's nodes.  
* **Dynamic Shader Generation:** The fragment shader for the force calculation is not static. It's a string template that's programmatically generated on the CPU based on the number of quadtree levels. This allows the shader to efficiently traverse the quadtree data.  
* **Multi-Pass Pipeline:** The simulation is broken down into a series of rendering passes:  
  1. **Clear Passes:** Clear the quadtree level textures.  
  2. **Calculation Passes:** Render a point for each particle to calculate and aggregate data for each quadtree level. This uses a gl.POINTS primitive and a blend function (gl.add) to accumulate mass and center-of-mass data.  
  3. **Force Pass:** A final pass renders a quad that uses the dynamically generated shader to traverse the quadtree textures and calculate the forces.

#### **Practical Caveats & Pitfalls**

* **Mental Overhead:** The complexity of this approach is its biggest drawback. It's incredibly difficult to read, debug, and maintain.  
* **Debugging:** The dynamically generated shaders are a nightmare to debug. You can't just open a file; you have to inspect a generated string, which is prone to errors.  
* **Performance vs. Readability:** This approach prioritizes performance at all costs, sacrificing all notions of code clarity and maintainability.  
* **Unorthodox Practices:** The project uses a blend of regl and manual WebGL calls, along with other unconventional practices that make it a "menace" to work with.

---

## **Frameworks and Platforms**

### **WebGL vs. WebGL2**

* **WebGL (1.0):** This version would be a non-starter for our project. It lacks key features that we rely on:  
  * **No Multiple Render Targets (MRT):** We can't write to multiple textures at once.  
  * **No Floating-Point Textures:** Support for floating-point textures is an optional extension, not a core feature. We need them for accurate physics.  
  * **No Integer Textures:** We can't store integer data directly in textures.  
* **WebGL2 (The Choice):** We have correctly chosen WebGL2. It provides all the necessary features as core parts of the specification, including MRT, floating-point textures, and integer textures. It also introduces compute shaders.  
* **WebGPU (The Future):** This is a new API that is designed to provide better access to modern GPU hardware. It is still in development, but it will eventually replace WebGL2. It will provide a more powerful and flexible way to perform GPU-based simulations.

### **Libraries**

* **regl:** The "Precut" project uses regl, which is a functional WebGL wrapper that simplifies WebGL programming. While it makes code more concise, it can also lead to highly dense and unreadable code.  
* **Manual WebGL2:** Our current approach is to use pure WebGL2. This gives us full control and a deeper understanding of the underlying mechanics, but requires more boilerplate code.

## **The Way Forward: A Combined and Pragmatic Approach**

Based on our analysis, here is the recommended path forward:

1. **Embrace Textures:** We'll continue with the **Textures and Render-to-Texture** approach. It offers the best balance of performance, flexibility, and maintainability. It's a well-understood and widely supported pattern for GPU-based simulations.  
2. **Integrate Key Ideas from "Precut" (The Good Ones):** While the "Precut" project's overall implementation is a "menace," it contains some valuable ideas that we can adapt.  
   * **Pre-sorting (The "Precut" Phase):** We will implement a CPU-side pre-sorting phase for the particle data. We'll use a spatial sorting algorithm (like a Hilbert curve) to order the particles, improving data locality for neighbor finding. This will make the gl\_physics.js shader much more efficient.  
   * **Quadtree for Forces:** We will implement a quadtree-based force calculation, but we'll do so in a clean, maintainable way, rather than with dynamic shader generation. This will be a significant engineering challenge, but it's a worthwhile one.  
3. **Use Compute Shaders:** We will use **compute shaders** for the physics calculations, which is a more modern and appropriate approach than using a vertex shader and transform feedback or a fragment shader. WebGL2's compute shader extension is designed for exactly this type of general-purpose GPU computation.

By combining the best aspects of these approaches, we can build a robust, high-performance particle system that is also readable and maintainable.