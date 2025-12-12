### Image

- Docker Hub: [`sheze666/htfem-planeinclusions`](https://hub.docker.com/r/sheze666/htfem-planeinclusions)
- Recommended tag: `1.0` (or `latest` if not specified)

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



