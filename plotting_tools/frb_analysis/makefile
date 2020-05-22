CC=gcc -g
CFLAGS=-lm -lcpgplot
DEPS = sigproc.h header.h fold.h polyco.h epn.h version.h slamac.h slalib.h

OBJ = frb_analysis.o functions.o file_exists.o error_message.o strings_equal.o print_version.o \
	open_file.o help.o read_header.o nsamples.o open_log.o fold_data.o \
	write_profiles.o update_log.o read_polyco.o nrselect.o norm_prof.o \
	read_block.o fold_header.o send_stuff.o swap_bytes.o aliases.o mjd.o \
	write_epn.o scale_prof.o vmin.o sizeof_file.o slalib.o pack_unpack.o \
	vmax.o close_log.o colorIndex.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

frb_analysis: $(OBJ)
	gcc -o $@ $^ $(CFLAGS)

clean: 
	rm -rf *.o
	rm -rf *~
