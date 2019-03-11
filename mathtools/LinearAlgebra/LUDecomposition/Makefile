# For the LU Decomposition

# Compiler options
FC    :=  gfortran
FFLAGS:=  -O3 -g -Wall -Warray-bounds -ffixed-line-length-none -fbounds-check 
BINDIR:= build

#Main
main:=unitytest
#Modules
mod:=LU

# ALL Modules
OBJS := $(BINDIR)/$(mod).o  $(BINDIR)/$(main).o

$(main) : $(OBJS) | $(BINDIR) 
	$(FC) $(FFLAGS) -o $(BINDIR)/$(main) $(OBJS) 

$(BINDIR)/$(mod).o : $(mod).f90 | $(BINDIR) 
	$(FC) $(FFLAGS) -c $(mod).f90 -o $(BINDIR)/$(mod).o -J $(BINDIR)

$(BINDIR)/$(main).o : $(main).f90 $(mod).f90 | $(BINDIR)
	$(FC) $(FFLAGS) -c $(main).f90 -o $(BINDIR)/$(main).o -I $(BINDIR) 

$(BINDIR):
	mkdir -p $(BINDIR)

clean :
	rm -rf $(BINDIR)

run : $(BINDIR)/$(main)
	$(BINDIR)/./$(main)
