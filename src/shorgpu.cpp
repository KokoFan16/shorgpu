#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <tuple>
#include <memory>
#include <random>
#include <numeric>
#include <chrono>
#include <bitset>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <openacc.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <mpi.h>
#include <omp.h>

using namespace std::string_literals;

constexpr double pi       = 3.14159'26535'89793'23846'26433'83279'502884;
constexpr double invsqrt2 = 0.70710'67811'86547'52440'08443'62104'849039;

int mpi_rank = -1;
int mpi_size = 0;
int mpi_xrank = -1;
int mpi_xsize = 0;
uint64_t num_local = 0;  // must be 64 bit for oracle arithmetics
uint64_t global_qubits = 0;
uint64_t local_qubits = 0;
cuDoubleComplex* psi = nullptr;

std::string uuidstr(const char uuid[16]);
int error(std::string message,
          bool onlyrank0 = false);
uint64_t modular_inverse(uint64_t a,
                         uint64_t N);
uint64_t modular_multiply(uint64_t a,
                          uint64_t b,
                          uint64_t N);
double compute_norm();
void dump_psi(bool includezeros = true, const std::string title = "");

int main(int argc,
         char* argv[]) {
    using std::min;
    using std::max;

    // initialize MPI for distributed memory (xcomm: split x=0 and x=1, local_comm: try to use one local process per GPU)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    global_qubits = std::log2(mpi_size);
    if (mpi_size != (1 << global_qubits) || global_qubits < 1 || global_qubits > 31)
        return error("mpi_size = "+std::to_string(mpi_size)+" must be >= 2 and a power of 2 (global_qubits = "+std::to_string(global_qubits)+": need at least 1 and not more than 31 global qubits)", true);
    int mpi_x = (mpi_rank < mpi_size/2 ? 0 : 1);
    MPI_Comm mpi_xcomm;
    MPI_Comm_split(MPI_COMM_WORLD, mpi_x, mpi_rank, &mpi_xcomm);
    MPI_Comm_rank(mpi_xcomm, &mpi_xrank);
    MPI_Comm_size(mpi_xcomm, &mpi_xsize);
    if (mpi_rank != mpi_x*mpi_xsize + mpi_xrank)
        return error("mpi_rank != mpi_x*mpi_xsize + mpi_xrank (mpi_rank,mpi_size,mpi_xrank,mpi_xsize="+std::to_string(mpi_rank)+','+std::to_string(mpi_size)+','+std::to_string(mpi_xrank)+','+std::to_string(mpi_xsize)+')');
    int mpi_local_rank = -1;
    MPI_Comm mpi_local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank, MPI_INFO_NULL, &mpi_local_comm);
    MPI_Comm_rank(mpi_local_comm, &mpi_local_rank);
    MPI_Comm_free(&mpi_local_comm);
    if (acc_get_num_devices(acc_device_nvidia) != 0)
        acc_set_device_num(mpi_local_rank%acc_get_num_devices(acc_device_nvidia), acc_device_nvidia);
    #pragma acc init

    // parse arguments
    if (argc < 4)
        error("Usage: "s+argv[0]+" N a t [repeat=1] [seed=-1] [errorseed=-1] [append=1] [cpprandom=1] [saverandom=0] [buffercount=2**21] [delta=0] [quantumerrors=0] [bandwidth=-1] [outfile=classicalbits.out]", true);
    uint64_t N = std::stoull(argv[1]);
    uint64_t a = std::stoull(argv[2]);
    uint64_t t = std::stoull(argv[3]);
    if (int64_t(N) < 6)
        return error("N = "+std::to_string(int64_t(N))+" is too small", true);
    if (auto gcd = std::gcd(a,N); gcd != 1)
        return error("a = "+std::to_string(a)+" must be coprime to N = "+std::to_string(N)+" to be invertible (gcd="+std::to_string(gcd)+" != 1)", true);
    if (t < 1)
        return error("t = "+std::to_string(t)+" is too small", true);
    if (t >= 127)
        return error("t = "+std::to_string(t)+" is too large", true);
    uint64_t L = std::ceil(std::log2(N));
    uint64_t num_qubits = L + 1;
    local_qubits = num_qubits - global_qubits;
    num_local = 1L << local_qubits;
    if (global_qubits >= num_qubits)
        return error("Too many global qubits: need at least 1 local qubit (num_qubits,global_qubits="+std::to_string(num_qubits)+','+std::to_string(global_qubits)+')', true);
    if (local_qubits < 1 || local_qubits >= 32)  // we shouldn't have more than 30 local qubits = 16 GiB on each GPU anyway
        return error("Too many or zero local qubits: For more than 32 local qubits, modular multiplication and addressing in MPI send and recv will overflow (num_qubits,global_qubits="+std::to_string(num_qubits)+','+std::to_string(global_qubits)+')', true);
    uint64_t repeat = 1;
    int64_t seed = -1;
    int64_t errorseed = -1;
    bool append = true;
    bool cpprandom = true;
    bool saverandom = false;
    uint64_t buffercount = 1 << 21;  // see optimize-buffercount.png
    double delta = 0;
    int quantumerrors = 0;  // 0 -> classical errors, 1 -> quantum (measurement) error (see paper), 2 -> quantum amplitude initialization error (63.19), 3 -> quantum phase initialization error (63.19), TODO: 4 -> gate error with Gaussian noise on rotation gates,  any other value -> no error at all even if delta != 0
    int bandwidth = -1;  // to implement Coppersmith's banded version of the QFT (cut off all rotations with angle < pi/2^bandwidth , see 63.20); NOTE: bandwidth = m_Coppersmith = dmax_Fowler = bandwidth_Nam = bandparameter_Cai - 2
    std::string outfile = "classicalbits.out";
    bool debug = false;      // undocumented, a lot of output, only do this for small systems (initializes each amplitude to a unique number and dumps the state after the oracle gate to debug the mpi permutation)
    bool debugshor = false;  // undocumented, a lot of output, only do this for small systems (dumps the state after each operation)
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        auto pos = arg.find('=');
        if (arg.substr(0,pos) == "repeat")             repeat = std::stoul(arg.substr(pos+1));
        else if (arg.substr(0,pos) == "seed")          seed = std::stoll(arg.substr(pos+1));
        else if (arg.substr(0,pos) == "errorseed")     errorseed = std::stoll(arg.substr(pos+1));
        else if (arg.substr(0,pos) == "append")        append = std::stol(arg.substr(pos+1));
        else if (arg.substr(0,pos) == "cpprandom")     cpprandom = std::stol(arg.substr(pos+1));
        else if (arg.substr(0,pos) == "saverandom")    saverandom = std::stol(arg.substr(pos+1));
        else if (arg.substr(0,pos) == "buffercount")   buffercount = std::stoull(arg.substr(pos+1));
        else if (arg.substr(0,pos) == "delta")         delta = std::stod(arg.substr(pos+1));
        else if (arg.substr(0,pos) == "quantumerrors") quantumerrors = std::stoi(arg.substr(pos+1));
        else if (arg.substr(0,pos) == "bandwidth")     bandwidth = std::stoi(arg.substr(pos+1));
        else if (arg.substr(0,pos) == "outfile")       outfile = arg.substr(pos+1);
        else if (arg.substr(0,pos) == "debug")         debug = std::stol(arg.substr(pos+1));
        else if (arg.substr(0,pos) == "debugshor")     debugshor = std::stol(arg.substr(pos+1));
        else error("Unknown parameter: "+arg, true);
    }
    if (bandwidth == -1)
        bandwidth = t-1;  // in this case we do the full QFT

    // boundary checks on buffercount, MPI message tags, and number of requests
    if (mpi_rank == 0 && num_local/buffercount >= 16384)
        std::cout << "[WARN] buffercount=" << buffercount << " may be too small, look out for an MPI crash (num_local/buffercount=" << num_local/buffercount << " > 16384)" << std::endl;
    if (mpi_rank == 0 && buffercount*sizeof(cuDoubleComplex) >= INT_MAX)
        std::cout << "[WARN] buffercount=" << buffercount << " may be too large, look out for an MPI crash (buffercount*sizeof(cuDoubleComplex)=" << buffercount*sizeof(cuDoubleComplex) << " > INT_MAX=" << INT_MAX << ')' << std::endl;
    int mpi_flag = 0;
    int mpi_tag_ub = 0;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &mpi_tag_ub, &mpi_flag);
    if (uint64_t(mpi_tag_ub) < 2*(num_local-1)/buffercount+1)
        return error("buffercount="+std::to_string(buffercount)+" is too small: need message tags up to "+std::to_string(2*(num_local-1)/buffercount+1)+" but mpi_tag_ub is only "+std::to_string(mpi_tag_ub), true);
    const uint64_t mpi_max_requests = 4ULL*mpi_xsize*max<uint64_t>(num_local/buffercount,1);
    if(mpi_max_requests > INT_MAX)  // if this fails, we could alternatively do MPI_Waitall before it overflows... but so many messages should never be sent anyway
        return error("buffercount="+std::to_string(buffercount)+" is too small: would need "+std::to_string(mpi_max_requests)+" > INT_MAX="+std::to_string(INT_MAX)+" message requests", true);
    MPI_Request* mpi_requests = new MPI_Request[mpi_x*mpi_max_requests];

    // generate the random numbers used for the sampling in the measurement, and for quantumerrors=0,1 (must be used only by mpi_rank == mpi_xsize <=> mpi_x == 1 && mpi_xrank == 0 <=> |10...0***>, so use .at() for access)
    std::vector<double> random, random2;
    if (mpi_rank == mpi_xsize) {
        random.resize(repeat*t);
        random2.resize(repeat*t);
        if (cpprandom) {
            std::random_device seeder;
            if (seed < 0)
                seed = seeder();
            if (errorseed < 0)
                errorseed = seeder();
            std::default_random_engine generator(seed), errorgenerator(errorseed);
            std::uniform_real_distribution<double> distribution(0.0, 1.0);  // r is drawn from [0.0, 1.0)
            for (auto& r : random)
                r = distribution(generator);
            for (auto& r : random2)
                r = distribution(errorgenerator);
        } else {
            if (seed < 0)
                seed = std::time(nullptr);
            std::srand(seed);
            for (auto& r : random)
                r = 1.*std::rand() / RAND_MAX;
            if (errorseed < 0)
                errorseed = std::rand();
            std::srand(errorseed);
            for (auto& r : random2)
                r = 1.*std::rand() / RAND_MAX;
        }
        if (saverandom) {
            std::ofstream random_file("randomnumbers.out", append ? std::ios::app : std::ios::out);
            random_file.precision(16);
            for (auto r : random)
                random_file << r << std::endl;
            for (auto r : random2)
                random_file << r << std::endl;
        }
    }
    MPI_Bcast(&seed, 1, MPI_INT64_T, mpi_xsize, MPI_COMM_WORLD);  // mpi_rank = mpi_xsize will do the sampling but mpi_rank = 0 will print the seeds
    MPI_Bcast(&errorseed, 1, MPI_INT64_T, mpi_xsize, MPI_COMM_WORLD);

    // determine the number of unique GPUs used in this computation by collecting their UUIDs
    int cudadev = acc_get_device_num(acc_get_device_type());
    cudaDeviceProp deviceprop;
    cudaGetDeviceProperties(&deviceprop, cudadev);
    std::set<std::string> gpus;
    if (mpi_rank != 0) {
        MPI_Send(deviceprop.uuid.bytes, 16, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    } else {
        gpus.insert(uuidstr(deviceprop.uuid.bytes));
        for (int i = 1; i < mpi_size; ++i) {
            char uuid[16] {};
            MPI_Recv(uuid, 16, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            gpus.insert(uuidstr(uuid));
        }
    }

    // print status information
    if (mpi_rank == 0) {
        time_t caltime = std::time(nullptr);
        std::cout << "SHORGPU 1.0 (" << std::put_time(std::localtime(&caltime),"%c %Z") << ")" << std::endl;
        int driverversion, runtimeversion;
        cudaDriverGetVersion(&driverversion);
        cudaRuntimeGetVersion(&runtimeversion);
        std::cout << "[INFO] Total number of GPUs used: " << gpus.size() << std::endl;
        std::cout << "[INFO] CUDA device " << cudadev << ": " << deviceprop.name << " (UUID: " << uuidstr(deviceprop.uuid.bytes) << ")" << std::endl;
        std::cout << "[INFO] CUDA device " << cudadev << " version (driver / runtime / capability): "
            << driverversion/1000 << '.' << (driverversion%100)/10 << " / "
            << runtimeversion/1000 << '.' << (runtimeversion%100)/10 << " / "
            << deviceprop.major << '.' << deviceprop.minor << std::endl;
        std::cout << "[INFO] CUDA device " << cudadev << " properties (memory / clock rate / multiprocessors / threads per multiprocessor): "
            << deviceprop.totalGlobalMem/1024./1024./1024. << " GB / "
            << deviceprop.clockRate*1e-6 << " GHz / "
            << deviceprop.multiProcessorCount << " / "
            << deviceprop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "[INFO] OpenACC device type: " << acc_get_device_type() << std::endl;
        std::cout << "[INFO] OpenACC device used: " << acc_get_device_num(acc_get_device_type()) << std::endl;
        char mpi_processor[MPI_MAX_PROCESSOR_NAME] = {};
        int mpi_processor_len;
        MPI_Get_processor_name(mpi_processor, &mpi_processor_len);
        std::cout << "[INFO] MPI root processor: " << mpi_processor << std::endl;
        std::cout << "[INFO] MPI number of processes: " << mpi_size << std::endl;
        if (mpi_size != gpus.size())
            std::cout << "[WARN] Number of GPUs is not equal to the number of MPI processes, probably using over- or undersubscription" << std::endl;
        std::cout << "[INFO] Simulating Shor's algorithm with sc-FT on GPUs for N=" << N << " a=" << a << " t=" << t
                  << " [repeat=" << repeat << "] [seed=" << seed <<  "] [errorseed=" << errorseed << "] [append=" << append << "]"
                  << " [cpprandom=" << cpprandom << "] [saverandom=" << saverandom << "] [buffercount=" << buffercount << "]"
                  << " [delta=" << delta << "] [quantumerrors=" << quantumerrors << "] [bandwidth=" << bandwidth << "] [outfile=" << outfile << "]" << std::endl;
        std::cout << "[INFO] Allocating memory for psi and psibuf per GPU: 2*" << num_local*sizeof(cuDoubleComplex) << " B" << std::endl;
        std::cout << "[INFO] Number of global + local qubits: " << global_qubits << " + " << local_qubits << " = " << num_qubits << std::endl;
    }

    // allocate memory for each process: global qubits = MPI rank (29 local qubits: 2*16*2^29 B + 2*4*2^29 B = (16 + 4) GiB = 20 GiB per GPU => maximum on 2048=2^11 A100 GPUs: 40 qubits (so 40 TiB in total))
    cuDoubleComplex* psibuf = nullptr;
    int* oracle_idxsend = nullptr;
    int* oracle_idxrecv = nullptr;
    psi = new cuDoubleComplex[num_local]();
    #pragma acc enter data copyin(psi[0:num_local])
    cudaMalloc(&psibuf, num_local*sizeof(cuDoubleComplex));
    cudaMalloc(&oracle_idxsend, num_local*sizeof(int));
    cudaMalloc(&oracle_idxrecv, num_local*sizeof(int));
    int* oracle_countsend = new int[mpi_x*mpi_xsize]();
    int* oracle_countrecv = new int[mpi_x*mpi_xsize]();
    int* oracle_offsetsend = new int[mpi_x*mpi_xsize]();
    int* oracle_offsetrecv = new int[mpi_x*mpi_xsize]();

    // precompute a^(2^(t-1-cbit)) mod N and its inverse, overflow-aware
    uint64_t ainv = modular_inverse(a, N);
    uint64_t* apow = new uint64_t[t]();
    uint64_t* apowinv = new uint64_t[t]();
    for (int cbit = 0; cbit < t; ++cbit) {
        apow[cbit] = a;
        apowinv[cbit] = ainv;
        for (int i = 0; i < t-1-cbit; ++i) {
            apow[cbit] = modular_multiply(apow[cbit], apow[cbit], N);
            apowinv[cbit] = modular_multiply(apowinv[cbit], apowinv[cbit], N);
        }
    }

    // prepare output
    std::ofstream out;
    if (mpi_rank == 0)
        out.open(outfile, append ? std::ios::app : std::ios::out);

    // prepare timing (only mpi_xrank=0 will average and print timing because the mpi_x=0 processes don't do the oracle)
    std::ofstream timing_file;
    if (mpi_xrank == 0)
        timing_file.open("timing.txt", append ? std::ios::app : std::ios::out);
    enum timing_id : size_t {time_init, time_oraclecount, time_oraclearrange, time_oraclempi, time_oraclecopyrotate, time_hadamardmpi, time_hadamardgpu, time_measurement, time_reset, time_output, time_N};
    const char* timing_labels[time_N] = {"init", "oraclecount", "oraclearrange", "oraclempi", "oraclecopyrotate", "hadamardmpi", "hadamardgpu", "measurement", "reset", "output"};
    double timing_data[time_N] = {0.0};
    double timing_sum[time_N] = {0.0};
    double this_time, start_time;
    this_time = start_time = MPI_Wtime();
    auto time_it = [&this_time,&timing_data] (timing_id tid) {
        double new_time = MPI_Wtime();
        timing_data[tid] += new_time - this_time;
        this_time = new_time;
    };

    // main loop of scFT Shor algorithm
    for (uint64_t rep = 0; rep < repeat; ++rep) {
        // initialize in |+>|0...01> (unless in debug mode or quantum initialization error mode)
        if (debug) {
            #pragma acc parallel loop independent present(psi[0:num_local])
            for (uint64_t i = 0; i < num_local; ++i)
                psi[i] = {mpi_rank, mpi_xrank*num_local + i};
        } else if (mpi_rank == 0 || mpi_rank == mpi_xsize) {
            double amplitude_real = invsqrt2;
            double amplitude_imag = 0;
            if (quantumerrors == 2) {  // note that this will not have an effect on measuring j0 (which is always 50:50), but phases and amplitudes can survive in second register! (see 63.19 bottom)
                if (mpi_rank == 0)  // |00...0***>
                    amplitude_real *= std::sqrt(1 + delta);
                else if (mpi_rank == mpi_xsize)  // |10...0***>
                    amplitude_real *= std::sqrt(1 - delta);
            } else if (quantumerrors == 3) {  // 63.19
                if (mpi_rank == mpi_xsize) {  // |10...0***>
                    amplitude_real = invsqrt2 * std::cos(pi*delta);
                    amplitude_imag = invsqrt2 * std::sin(pi*delta);
                }
            }
            #pragma acc parallel loop independent present(psi[0:num_local])
            for (uint64_t i = 0; i < num_local; ++i)
                if (i == 1)  // |...01>
                    psi[i] = {amplitude_real,amplitude_imag};
                else
                    psi[i] = {0.,0.};
        } else {
            #pragma acc parallel loop independent present(psi[0:num_local])
            for (uint64_t i = 0; i < num_local; ++i)
                psi[i] = {0., 0.};
        }
        if (debugshor)
            dump_psi(false, ">>>INIT<<<");
        time_it(time_init);

        // loop over the cbits j = jt-1 ... j0 to measure (each iteration measures jcbit); j = jhi * 2^64 + jlo (NOTE: j is constrained to 128 bits!)
        uint64_t jhi = 0;
        uint64_t jlo = 0;
        uint64_t bmaskhi = 0;  // for banded QFT: take only <bandwidth> most-significant bits of j
        uint64_t bmasklo = 0;
        for (int cbit = 0; cbit < t; ++cbit) {
            // implementation of the unitary map for the oracle gate (controlled gate: only apply if mpi_x=1)
            if (mpi_x) {
                // first count how many amplitudes we have to send to the other GPUs and how many amplitudes we receive from other GPUs
                uint64_t a = apow[cbit];
                uint64_t ainv = apowinv[cbit];
                for (int other_xrank = 0; other_xrank < mpi_xsize; ++other_xrank) {
                    oracle_countsend[other_xrank] = 0;
                    oracle_countrecv[other_xrank] = 0;
                }
                #pragma acc parallel loop copy(oracle_countsend[0:mpi_xsize],oracle_countrecv[0:mpi_xsize])
                for (uint64_t i = 0; i < num_local; ++i) {
                    uint64_t y = mpi_xrank*num_local + i;
                    uint64_t ysend = (y < N ? modular_multiply(a,y,N) : y);     // U |y> = |ysend>
                    uint64_t yrecv = (y < N ? modular_multiply(ainv,y,N) : y);  // U |yrecv> = |y>
                    #pragma acc atomic update
                    ++oracle_countsend[ysend / num_local];
                    #pragma acc atomic update
                    ++oracle_countrecv[yrecv / num_local];
                }
                oracle_offsetsend[0] = 0;
                oracle_offsetrecv[0] = 0;
                for (int other_xrank = 1; other_xrank < mpi_xsize; ++other_xrank) {
                    oracle_offsetsend[other_xrank] = oracle_offsetsend[other_xrank-1] + oracle_countsend[other_xrank-1];
                    oracle_offsetrecv[other_xrank] = oracle_offsetrecv[other_xrank-1] + oracle_countrecv[other_xrank-1];
                }
                time_it(time_oraclecount);

                // copy psi to psibuf and then prepare psi as source buffer for the contiguous memory transfer (63.11)
                #pragma acc parallel loop independent present(psi[0:num_local]) deviceptr(psibuf)
                for (uint64_t i = 0; i < num_local; ++i)
                    psibuf[i] = psi[i];
                #pragma acc parallel loop present(psi[0:num_local]) deviceptr(psibuf,oracle_idxsend) copyin(oracle_offsetsend[0:mpi_xsize])  // NOTE: oracle_offsetsend will be atomically modified and must not be copied back to the host
                for (uint64_t i = 0; i < num_local; ++i) {
                    uint64_t y = mpi_xrank*num_local + i;
                    uint64_t ysend = (y < N ? modular_multiply(a,y,N) : y);  // U |y> = |ysend>
                    int other_xrank = ysend / num_local;
                    int idx = 0;
                    #pragma acc atomic capture
                    idx = oracle_offsetsend[other_xrank]++;
                    psi[idx] = psibuf[i];
                    oracle_idxsend[idx] = ysend % num_local;
                }
                time_it(time_oraclearrange);

                // MPI transfer psi from this GPU -> psibuf from other GPUs (post recvs first, and take care of well-ordered sends; this version was faster than MPI_Alltoallv)
#ifdef ALLTOALL
                #pragma acc host_data use_device(psi)
                MPI_Alltoallv(psi, oracle_countsend, oracle_offsetsend, MPI_C_DOUBLE_COMPLEX,
                              psibuf, oracle_countrecv, oracle_offsetrecv, MPI_C_DOUBLE_COMPLEX,
                              mpi_xcomm);
                MPI_Alltoallv(oracle_idxsend, oracle_countsend, oracle_offsetsend, MPI_INT,
                              oracle_idxrecv, oracle_countrecv, oracle_offsetrecv, MPI_INT,
                              mpi_xcomm);
#else
                int num_requests = 0;
                for (int delta_xrank = 0; delta_xrank < mpi_xsize; ++delta_xrank) {
                    int recv_xrank = (mpi_xrank + delta_xrank) % mpi_xsize;
                    for (int sent = 0; sent < oracle_countrecv[recv_xrank]; sent += min<int>(oracle_countrecv[recv_xrank]-sent,buffercount)) {
                        MPI_Irecv(psibuf         + oracle_offsetrecv[recv_xrank] + sent, min<int>(oracle_countrecv[recv_xrank]-sent,buffercount), MPI_C_DOUBLE_COMPLEX, recv_xrank, 2*(sent / buffercount),   mpi_xcomm, &mpi_requests[num_requests++]);
                        MPI_Irecv(oracle_idxrecv + oracle_offsetrecv[recv_xrank] + sent, min<int>(oracle_countrecv[recv_xrank]-sent,buffercount), MPI_INT,              recv_xrank, 2*(sent / buffercount)+1, mpi_xcomm, &mpi_requests[num_requests++]);
                    }
                    int send_xrank = (mpi_xrank - delta_xrank + mpi_xsize) % mpi_xsize;
                    for (int sent = 0; sent < oracle_countsend[send_xrank]; sent += min<int>(oracle_countsend[send_xrank]-sent,buffercount)) {
                        #pragma acc host_data use_device(psi)
                        MPI_Isend(psi            + oracle_offsetsend[send_xrank] + sent, min<int>(oracle_countsend[send_xrank]-sent,buffercount), MPI_C_DOUBLE_COMPLEX, send_xrank, 2*(sent / buffercount),   mpi_xcomm, &mpi_requests[num_requests++]);
                        MPI_Isend(oracle_idxsend + oracle_offsetsend[send_xrank] + sent, min<int>(oracle_countsend[send_xrank]-sent,buffercount), MPI_INT,              send_xrank, 2*(sent / buffercount)+1, mpi_xcomm, &mpi_requests[num_requests++]);
                    }
                }
                MPI_Waitall(num_requests, mpi_requests, MPI_STATUS_IGNORE);
#endif
                time_it(time_oraclempi);

                // copy and rearrange psibuf -> psi (63.11) and implement the R gates while doing that
                double phi = pi*(std::ldexp(jhi&bmaskhi,-cbit+64) + std::ldexp(jlo&bmasklo,-cbit));  // pi*j / 2^cbit = pi*(jhi * 2^64 + jlo) / 2^cbit = pi * 0.jcbit-1...j0 (63.10); banded QFT: take only the <bandwidth> most-significant bits
                double c = std::cos(phi);
                double s = std::sin(phi);
                #pragma acc parallel loop independent present(psi[0:num_local]) deviceptr(psibuf,oracle_idxrecv)
                for (uint64_t i = 0; i < num_local; ++i) {
                    cuDoubleComplex tmp = psibuf[i];
                    psi[oracle_idxrecv[i]].x = c*tmp.x - s*tmp.y;
                    psi[oracle_idxrecv[i]].y = c*tmp.y + s*tmp.x;
                }
                time_it(time_oraclecopyrotate);
            }
            if (debug)
                dump_psi();
            if (debugshor)
                dump_psi(false, ">>>ORACLE<<< cbit="+std::to_string(cbit));

            // implementation of the H gate
            for (uint64_t sent = 0; sent < num_local; sent += min(num_local-sent,buffercount)) {
                #pragma acc host_data use_device(psi)
                MPI_Sendrecv(psi + sent,    min(num_local-sent,buffercount), MPI_C_DOUBLE_COMPLEX, mpi_rank ^ mpi_xsize, sent / buffercount,
                             psibuf + sent, min(num_local-sent,buffercount), MPI_C_DOUBLE_COMPLEX, mpi_rank ^ mpi_xsize, sent / buffercount,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            time_it(time_hadamardmpi);
            if (mpi_x) {
                // psibuf = psi_0*...*, psi = psi_1*...*
                #pragma acc parallel loop independent present(psi[0:num_local]) deviceptr(psibuf)
                for (uint64_t i = 0; i < num_local; ++i) {
                    psi[i].x = (psibuf[i].x - psi[i].x) * invsqrt2;
                    psi[i].y = (psibuf[i].y - psi[i].y) * invsqrt2;
                }
            } else {
                // psibuf = psi_1*...*, psi = psi_0*...*
                #pragma acc parallel loop independent present(psi[0:num_local]) deviceptr(psibuf)
                for (uint64_t i = 0; i < num_local; ++i) {
                    psi[i].x = (psibuf[i].x + psi[i].x) * invsqrt2;
                    psi[i].y = (psibuf[i].y + psi[i].y) * invsqrt2;
                }
            }
            if (debugshor)
                dump_psi(false, ">>>HADAMARD<<< cbit="+std::to_string(cbit));
            time_it(time_hadamardgpu);

            // implementation of the measurement (only mpi_rank == mpi_xsize has the result) and broadcast the measurement result from mpi_rank = mpi_xsize (0b10...0)
            double prob = 0;
            int jcbit = 0;
            int jcbitobserved = 0;
            if (mpi_x) {
                double p1 = 0.;
                #pragma acc parallel loop reduction(+:p1) present(psi[0:num_local])
                for (uint64_t i = 0; i < num_local; ++i) {
                    p1 += psi[i].x * psi[i].x;
                    p1 += psi[i].y * psi[i].y;
                }
                if (mpi_xrank == 0) {   // mpi_rank == mpi_xsize <=> mpi_x == 1 && mpi_xrank == 0 <=> |10...0***>
                    MPI_Reduce(MPI_IN_PLACE, &p1, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_xcomm);
                    double R = random.at(rep*t+cbit);
                    double R2 = random2.at(rep*t+cbit);
                    if (quantumerrors == 1 && delta != 0) {  // quantum measurement error
                        double p0 = abs(1 - p1);
                        double p0prime = (1 - delta)*p0 + delta*p1;
                        double p1prime = (1 - delta)*p1 + delta*p0;
                        double perror = 0;
                        if (R < p1prime) {
                            jcbitobserved = 1;
                            perror = delta*p0 / p1prime;
                        } else {
                            jcbitobserved = 0;
                            perror = delta*p1 / p0prime;
                        }
                        if (R2 < perror) {
                            jcbit = 1 - jcbitobserved;
                            prob = (jcbit == 0 ? p0 : p1);
                            std::cout << "[INFO] Quantum error happened: repetition=" << rep << " cbit=" << cbit << " jcbitobserved=" << jcbitobserved << " p1=" << p1 << " p1prime=" << p1prime << " R=" << R << " perror=" << perror << " R2=" << R2 << " prob=" << prob << std::endl;
                        } else {
                            jcbit = jcbitobserved;
                            prob = (jcbit == 0 ? p0 : p1);
                        }
                    } else {  // normal measurement or classical error
                        if (R < p1) {
                            jcbit = 1;
                            prob = p1;
                        } else {
                            prob = abs(1 - p1);
                        }
                        jcbitobserved = jcbit;
                        if (quantumerrors == 0 && delta != 0 && R2 < delta) {  // classical error
                            jcbitobserved = 1 - jcbit;
                            std::cout << "[INFO] Classical error happened: repetition=" << rep << " cbit=" << cbit << " jcbitobserved=" << jcbitobserved << std::endl;
                        }
                    }
                } else {
                    MPI_Reduce(&p1, 0, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_xcomm);
                }
            }
            MPI_Bcast(&prob, 1, MPI_DOUBLE, mpi_xsize, MPI_COMM_WORLD);
            MPI_Bcast(&jcbit, 1, MPI_INT, mpi_xsize, MPI_COMM_WORLD);
            MPI_Bcast(&jcbitobserved, 1, MPI_INT, mpi_xsize, MPI_COMM_WORLD);
            if (jcbitobserved) {
                if (cbit >= 64)
                    jhi ^= (1ULL << (cbit-64));
                else
                    jlo ^= (1ULL << cbit);
            }
            bmaskhi = (bmaskhi << 1) | (bmasklo >> 63);
            bmasklo = (bmasklo << 1) | (cbit <= bandwidth ? 0b1 : 0b0);
            if (debugshor)
                dump_psi(false, ">>>MEASUREMENT<<< cbit="+std::to_string(cbit));
            time_it(time_measurement);

            // all mpi_x = jcbit copy all their amplitudes with MPI_set to all mpi_x = 1-jcbit, divide by sqrt(2*prob) for normalization
            if (mpi_x == jcbit) {
                double norm = 1./std::sqrt(2*prob);
                #pragma acc parallel loop independent present(psi[0:num_local])
                for (uint64_t i = 0; i < num_local; ++i) {
                    psi[i].x *= norm;
                    psi[i].y *= norm;
                }
                for (uint64_t sent = 0; sent < num_local; sent += min(num_local-sent,buffercount)) {
                    #pragma acc host_data use_device(psi)
                    MPI_Send(psi + sent, min(num_local-sent,buffercount), MPI_C_DOUBLE_COMPLEX, mpi_rank ^ mpi_xsize, sent / buffercount, MPI_COMM_WORLD);
                }
            } else {
                for (uint64_t sent = 0; sent < num_local; sent += min(num_local-sent,buffercount)) {
                    #pragma acc host_data use_device(psi)
                    MPI_Recv(psi + sent, min(num_local-sent,buffercount), MPI_C_DOUBLE_COMPLEX, mpi_rank ^ mpi_xsize, sent / buffercount, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            if (quantumerrors == 2) {  // if necessary, renormalize the amplitudes to initialize the respective quantumerrors case (63.19); here, amplitude mass is shifted from |1>|...> to |0>|...>
                double renormalize = (mpi_x == 0 ? std::sqrt(1 + delta) : std::sqrt(1 - delta));
                #pragma acc parallel loop independent present(psi[0:num_local])
                for (uint64_t i = 0; i < num_local; ++i) {
                    psi[i].x *= renormalize;
                    psi[i].y *= renormalize;
                }
            } else if (quantumerrors == 3) {  // here, all |1>|...> amplitudes are multiplied by e^(i*pi*delta)
                if (mpi_x == 1) {
                    double c = std::cos(pi*delta);
                    double s = std::sin(pi*delta);
                    #pragma acc parallel loop independent present(psi[0:num_local])
                    for (uint64_t i = 0; i < num_local; ++i) {
                        cuDoubleComplex tmp = psi[i];
                        psi[i].x = c*tmp.x - s*tmp.y;
                        psi[i].y = c*tmp.y + s*tmp.x;
                    }
                }
            }
            if (debugshor)
                dump_psi(false, ">>>RESET<<< cbit="+std::to_string(cbit));
            time_it(time_reset);
        }

        // output measured classical bitstring to file, and after 10 repetitions or at the end, accumulate and print timing data; also, check norm
        double norm = compute_norm();
        if (mpi_rank == 0) {
            out << (std::bitset<64>(jhi).to_string() + std::bitset<64>(jlo).to_string()).substr(128-t)
                << '\t' << (jhi ? std::to_string(jhi)+"*2^64+" : "") << jlo << std::endl;
            if (std::abs(1 - norm) > 1e-6)
                std::cout << "[WARN] Norm of psi after the simulation for repetition=" << rep << " is too far from 1.0: norm=" << std::setprecision(15) << norm << std::setprecision(6) << std::endl;
        }
        if (rep == 0 || rep == 9 || rep == repeat-1) {
            if (mpi_x) {  // only the mpi_x=1 processes do the timing because of the oracle
                MPI_Reduce(timing_data, timing_sum, time_N, MPI_DOUBLE, MPI_SUM, 0, mpi_xcomm);
                if (mpi_xrank == 0) {
                    timing_file << "Detailed timing information after " << rep+1 << " repetitions (averaged over all mpi_x=1 processes):\n";
                    for (size_t tid = time_init; tid < time_N; ++tid) {
                        timing_file << '\t' << std::setw(20) << std::left << timing_labels[tid] << std::setw(20) << std::fixed << std::setprecision(10) << timing_sum[tid]/mpi_xsize;
                        if (time_init < tid && tid < time_output)
                            timing_file << std::setw(30) << timing_labels[tid]+"/repeat/t"s << std::setw(20) << timing_sum[tid]/mpi_xsize/(rep+1)/t << std::endl;
                        else
                            timing_file << std::setw(30) << timing_labels[tid]+"/repeat"s << std::setw(20) << timing_sum[tid]/mpi_xsize/(rep+1) << std::endl;
                    }
                }
            }
        }
        time_it(time_output);
    }

    // print total elapsed time
    if (mpi_rank == 0)
        std::cout << "[INFO] Total simulation time: " << MPI_Wtime()-start_time << std::endl;

    // free resources
    delete[] apowinv;
    delete[] apow;
    delete[] oracle_offsetrecv;
    delete[] oracle_offsetsend;
    delete[] oracle_countrecv;
    delete[] oracle_countsend;
    delete[] mpi_requests;
    if (oracle_idxrecv) cudaFree(oracle_idxrecv);
    if (oracle_idxsend) cudaFree(oracle_idxsend);
    if (psibuf) cudaFree(psibuf);
    if (psi) delete[] psi;
    if (mpi_xcomm) MPI_Comm_free(&mpi_xcomm);
    MPI_Finalize();
    return 0;
}

std::string uuidstr(const char uuid[16]) {
    std::ostringstream ostr;
    ostr << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i)
        ostr << std::setw(2) << (0xff & uuid[i])
             << (i == 3 || i == 5 || i == 7 || i == 9 ? "-" : "");
    return ostr.str();
}

int error(std::string message, bool onlyrank0) {
    if (onlyrank0 && mpi_rank != 0)
        return -1;
    std::cerr << "Error from rank "+std::to_string(mpi_rank)+": "+message << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
    return -1;
}

inline uint64_t modular_inverse(uint64_t a,
                                uint64_t N) {
    // given that gcd(a,N) = a*b + N*M = 1, compute b = a^-1 mod N using the extended Euclidean algorithm
    using std::tie;
    using std::make_tuple;
    int64_t b = 0, newb = 1;
    int64_t g = N, newg = a;  // g will be gcd(a,N) = 1
    while (newg != 0) {
        int64_t quotient = g / newg;
        tie(b, newb) = make_tuple(newb, b - quotient * newb);
        tie(g, newg) = make_tuple(newg, g - quotient * newg);
    }
    if (b < 0)
        b += N;
    return b;
}

inline uint64_t modular_multiply(uint64_t a,
                                 uint64_t b,
                                 uint64_t N) {
    // first try the multiplication and check for overflow; only if necessary, do the slower overflow-free multiplication
    // NOTE: since NVHPC 21.7, __int128 is supported, so in the future we might not need to emulate it ourselves here
    a %= N;
    b %= N;
    uint64_t res = a*b;
    if (a == 0 || res / a == b)
        return res % N;
    if (a > b) {  // swap
        res = a;
        a = b;
        b = res;
    }
    res = 0;
    while (b) {
        if (b & 0b1)
            res = (res + a) % N;
        a = (a * 2) % N;
        b >>= 1;
    }
    return res;
}

double compute_norm() {
    double norm = 0.;
    #pragma acc parallel loop reduction(+:norm) present(psi[0:num_local])
    for (uint64_t i = 0; i < num_local; ++i) {
        norm += psi[i].x * psi[i].x;
        norm += psi[i].y * psi[i].y;
    }
    if (mpi_rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&norm, 0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    return norm;
}

void dump_psi(bool includezeros, const std::string title) {
    std::cout << std::unitbuf;
    #pragma acc update self(psi[0:num_local])
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi_rank == 0)
        std::cout << title << std::endl;
    for (int r = 0; r < mpi_size; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == mpi_rank) {
            std::string global = std::bitset<64>(mpi_rank).to_string().substr(64-global_qubits);
            std::cout << "MPI rank r = " << r << " = 0b" << global << ':' << std::endl;
            for (uint64_t i = 0; i < num_local; ++i)
                if (includezeros || (psi[i].x*psi[i].x + psi[i].y*psi[i].y > 1e-10))
                    std::cout << "\t0b " << global << ' ' << std::bitset<64>(i).to_string().substr(64-local_qubits) << ' ' << psi[i].x << ' ' << psi[i].y << std::endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
