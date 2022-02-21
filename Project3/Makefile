ALL = cannons dns

all: $(ALL)

cannons: cannons.c
	mpicc -o cannons cannons.c -lm
dns: dns.c
	mpicc -o dns dns.c -lm
