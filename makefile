MPICPP ?= mpic++
NVCCFLAGS ?= -g -lineinfo
NVHPCFLAGS ?= -std=c++17 -acc -fast -gopt -gpu=lineinfo -Minfo=accel -cuda -Mint128
NVHPCDEFS ?= -DEMULATE_INT128
NVHPCINCS ?=
NVHPCLIBS ?=
EXE ?= shorgpu

# if NCCL is not found, set NCCL_HOME to point to your NCCL installation
# (can usually be found within you NVHPC installation directory)
ifneq ($(NCCL_HOME), "")
NVHPCINCS += -I$(NCCL_HOME)/include/
NVHPCLIBS += -L$(NCCL_HOME)/lib
endif
NVHPCLIBS += -lnccl

all: $(EXE:%=build/%)

build/%: obj/%.o
	$(MPICPP) $(NVHPCFLAGS) $(NVHPCINCS) $^ -o $@ $(NVHPCLIBS)

obj/%.o: src/%.cpp
	$(MPICPP) $(NVHPCFLAGS) $(NVHPCINCS) $(NVHPCDEFS) -c $< -o $@

clean:
	$(RM) $(EXE:%=build/%) $(EXE:%=obj/%.o)

rebuild:
	$(MAKE) clean
	$(MAKE)

.PHONY: clean rebuild
.PRECIOUS: $(wildcard src/*)
.SECONDARY:
