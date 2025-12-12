### Image

- Docker Hub: [`sheze666/htfem-planeinclusions`](https://hub.docker.com/r/sheze666/htfem-planeinclusions)
- Recommended tag: `1.0` 
### 1. Pull the image

```bash
docker pull sheze666/htfem-planeinclusions:1.0
```

### 2. Prepare input files

Create a working directory on your machine and put the required input files in it, e.g.

```text
/pathToCase
├── input_file.dat     # main input file (mesh, materials, loads, etc.)
└── points_xy.dat      # sampling points (x, y), one point per line
```

### 3. run with Docker (Linux/macOS)

```bash
cd /pathToCase
docker run --rm -v "$PWD":/work sheze666/htfem-planeinclusions:1.0
```

Explanation:
	• -v "$PWD":/work mounts the current directory on the host to /work inside the container.
	• The solver runs in /work and reads input_file.dat and points_xy.dat using relative paths.
	• output file (e.g. output_file.dat) are written back to the same host directory.

### 4. Run with Docker (Windows PowerShell)

```powershell
cd path
docker run --rm -v "${PWD}:/work" sheze666/htfem-planeinclusions:1.0
```


# HT-FEM Plane-Inclusion Input File: Format and Explanation

## 1. Overall Structure of `input_file.dat`

The input file is a plain text file divided into several blocks. Each block starts with a header line beginning with three asterisks (`***`). The blocks must appear in the following order:

1. Control parameters  
2. Element connectivity  
3. Node coordinates  
4. Constrained boundary conditions  
5. Material properties  
6. Load conditions  
7. Center of a circle  

Within each block, the data format must not be changed, otherwise the Fortran code will not be able to read the file correctly.

---

## 2. Control Parameters Block (`***Control parameters`)

**Example:**

~~~text
***Control parameters
NTYPE  NPOIN  NELEM  NVFIX
    1   1205     52     50
NDOFN  NDIME  NSTRE
    2      2      3
NMATS  NPROP  NGAUS
    2      5     11
NODEG
    5
~~~

With

- `NTYPE`: Analysis type (e.g., 1 = plane stress; This version does not include plane-stress problems.).
- `NPOIN`: Total number of nodes in the mesh. Must match the number of lines in the Node coordinates block.
- `NELEM`: Total number of elements. Must match the number of elements in the Element connectivity block.
- `NVFIX`: Number of constrained boundary conditions (number of BC lines in the Constrained boundary conditions block).
- `NDOFN`: Degrees of freedom per node (2 for `u1`, `u2`).
- `NDIME`: Spatial dimension (2 for 2D problems. This version only include 2D problems).
- `NSTRE`: Number of stress components (3 for σ11, σ22, σ12 in 2D).

- `NMATS`: Number of material sets defined in the Material properties block.
- `NPROP`: Number of properties for each material set (here 5 values per material line).
- `NGAUS`: Number of Gaussian points per element used in numerical integration.
- `NODEG`: Number of nodes per edge. Here `NODEG = 5` means each edge is described by 5 nodes along the side.

---

## 3. Element Connectivity Block (`***Element connectivity`)

**Example of the header line for an element:**

~~~text
Elem#   nside    Mat#    CornerFlag      x0       y0                 r0   nextline(Nod1# -------------> NodN#)
    1	   12	    2	          0       0        0        0.531808467
483	494	496	...
~~~

The element block uses two logical lines per element:

1. **Element header:**
   - `Elem#`: Element ID (1 to `NELEM`).
   - `nside`: Number of edges (sides) of the polygonal element.
   - `Mat#`: Material ID for this element (1..`NMATS`).
   - `CornerFlag`: 1 if the element is a corner element; 0 otherwise. (In this version, it must be set to 0.)
   - `x0, y0`: It only takes effect for corner elements; in this version, it is set to 0.
   - `r0`: The radius of the circular inclusion or hole.

2. **Node list (the next line):**
   - A sequence of node IDs (integers) defining the element boundary in counter-clockwise order.
   - The number of node IDs in this line must be consistent with the internal definition of the element and the boundary interpolation (related to `(NODEG-1) * sides`).

---

## 4. Node Coordinates Block (`***Node coordinates`)

**Header:**

~~~text
***Node coordinates
Node#     xCoord       yCoord
1	          -4	      -18
2	      -4.343	      -18
...
~~~

For each node `i` from 1 to `NPOIN`, one line is given:

- `Node#`: Global node index (1..`NPOIN`). This must be unique and consecutive.
- `xCoord`: x-coordinate of the node in the global Cartesian system.
- `yCoord`: y-coordinate of the node.

The node IDs used in the Element connectivity block must match the IDs listed here.

---

## 5. Constrained Boundary Conditions Block (`***Constrainted boundary conditions`)

**Header:**

~~~text
***Constrainted boundary conditions
BC#  Node#     (ifpre(i, idofn), idofn=1, ndofn)    (presc(i, idofn), idofn=1, ndofn)
1	     1	      1	  1                             0   0
2	     2	      1	  1                             0   0
...
~~~

There are `NVFIX` lines, one for each boundary condition:

- `BC#`: Boundary condition index (1..`NVFIX`).
- `Node#`: Node ID where the displacement is prescribed.
- `ifpre(i,1)`, `ifpre(i,2)`: Flags for each DOF. `1` means this DOF is fixed (prescribed); `0` means free.
- `presc(i,1)`, `presc(i,2)`: Prescribed displacement values for each DOF (usually 0 for fixed boundaries).

In the given example, the left and right vertical boundaries are fully clamped: both `u1` and `u2` are set to zero.

---

## 6. Material Properties Block (`***Material properties`)

**Header:**

~~~text
***Material properties
 Mat#        E_i           v_i          E_m       v_m     Thick
1        70000.0          0.22         2100      0.37       1.0
2            0.0          0.22         2100      0.37       1.0
~~~

For each material set (1..`NMATS`), one line with `NPROP` values is given. The exact meaning of the 5 properties is:

- `Mat#`: Material ID, referred to by the `Mat#` field in the Element connectivity block.
- Property 1 (`E_i`): Young’s modulus of inclusion.
- Property 2 (`v_i`): Poisson’s ratio of inclusion.
- Property 3 (`E_m`): Young’s modulus of matrix.
- Property 4 (`v_m`): Poisson’s of matrix.
- Property 5 (`Thick`): 1.0.

If `E_i = 0 MPa`, it denotes a hole.

---

## 7. Load Conditions Block (`***Load conditions`)

**Header:**

~~~text
***Load conditions (including the secend and third conditions)
NEDGE
  11
Edge#  Elem#                 Nod1#-->NODEG                   (press: normal into element; tangential positive CCW)
1	      34	      799   786   781   775   769	                80  0     80  0    80  0     80  0      80  0
...
~~~

Meaning:

- `NEDGE`: Number of loaded edges (boundary segments).

For each edge (`Edge# = 1..NEDGE`):

- `Elem#`: Element ID to which the loaded edge belongs.
- `Nod1#..NODEG`: Node IDs along that element edge, in order. The number of node IDs equals `NODEG`.
- Then, for each of these nodes, a pair of load components `(t_n, t_t)` is given:
  - `t_n`: Normal traction (positive when acting into the element interior).
  - `t_t`: Tangential traction (positive in the counter-clockwise direction along the boundary).

The solver interpolates these nodal tractions to obtain the distributed boundary load on each edge.

---

## 8. Center of a Circle Block (`***Center of a circle`)

**Header:**

~~~text
***Center of a circle
NcenEle
  52
Elem#     NO.line	                coord(1)---->(2)
1	           1	       -4.411420544	 6.663253698
2	           2	        4.446337019	 2.981980384
...
~~~

This block defines the centers of the inclusions or holes for each element:

- `NcenEle`: Number of elements that have inclusions or holes. 
- For each element:
  - `Elem#`: Element ID.
  - `NO.line`: Line Number.
  - `coord(1)`, `coord(2)`: Coordinates (`x_c`, `y_c`) of the center of the inclusion or hole.

---

## 9. How to Modify or Create a New Input File

When preparing a new example, you must keep the structure and format of `input_file.dat` unchanged:

1. Update `NPOIN`, `NELEM`, `NVFIX`, `NMATS`, etc., in the Control parameters block to match your new mesh.  
2. Provide the Element connectivity block for all elements (1..`NELEM`), with correct `Mat#` and (`r0`).  
3. List all node coordinates for node IDs 1..`NPOIN` in the Node coordinates block.  
4. Specify boundary conditions in the Constrained boundary conditions block, with exactly `NVFIX` lines.  
5. Define material data for all materials, each line containing exactly `NPROP` values.  
6. Describe boundary loads in the Load conditions block with `NEDGE` edges and `NODEG` nodes per edge, with traction components (`t_n`, `t_t`) at each node.  
7. Provide all centers of inclusions or holes.

As long as the overall block order and the header lines are preserved, you may change the numerical values to build different test cases (different geometries, material properties, loadings, and boundary conditions).



