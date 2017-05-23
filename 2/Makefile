LDFLAGS := -lGL -lGLU -lglut 

mandelbrot: mandelbrot.c
	$(CC) $^ -O3 -fopenmp -o $@ $(LDFLAGS) 
