# shorgpu: Large-Scale Simulation of Shor's Quantum Factoring Algorithm

`shorgpu` is a quantum computer simulator designed to simulate the iterative Shor algorithm on multiple GPUs. 

```bash
Usage: mpirun -n numGPUs build/shorgpu N a t [repeat=1] [seed=-1] [errorseed=-1] [append=1] [cpprandom=1] [saverandom=0] [buffercount=2**21] [delta=0] [quantumerrors=0] [outfile=classicalbits.out]
```

The terminology, quantum gates, and quantum errors implemented in the simulator are documented in the corresponding paper:

[Large-Scale Simulation of Shor's Quantum Factoring Algorithm](https://doi.org/10.3390/math11194222) <br>
_Dennis Willsch, Madita Willsch, Fengping Jin, Hans De Raedt, and Kristel Michielsen_ <br>
**Mathematics** 11, 4222 (2023)
