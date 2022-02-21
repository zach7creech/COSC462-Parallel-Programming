ALL = prefix_polynomial integration

all: $(ALL)

prefix_polynomial: prefix_polynomial.c
	mpicc -o prefix_polynomial prefix_polynomial.c -lm
integration: integration.c
	mpicc -o integration integration.c
