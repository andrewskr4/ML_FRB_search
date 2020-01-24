CC = gcc -O2
F77 = gfortran
LOADLIBES= -L /usr/lib64/ -lfftw3 -lm
CFLAGS = -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

SRCS= fil2spec.c strings_equal.c print_version.c \
	filterbank_header.c send_stuff.c writeooty2fb.c \
	swap_bytes.c read_header.c functions.c

HDRS=filterbank.h header.h epn.h polyco.h sigproc.h version.h

OBJ_FIL2SPEC= fil2spec.o strings_equal.o \
	print_version.o filterbank_header.o send_stuff.o \
	writeooty2fb.o swap_bytes.o read_header.o functions.o

fil2spec : $(OBJ_FIL2SPEC) $(HDRS)
	$(CC) $(CFLAGS) -o  fil2spec $(OBJ_FIL2SPEC)  $(LOADLIBES)
clean:
	rm *.o
	rm *~
