q: 
	gcc qwen25.c -O3 -ffast-math -g -lm -march=znver4 -mavx512bf16 -o qwen #   -fsanitize=address # -fopenmp  #  #
	gcc qwen25.c -O0 -ffast-math -g -lm -march=znver4 -mavx512bf16 -o qwend  -fsanitize=address # -fopenmp  # -fsanitize=address #

all:
	gcc gpt2.c -o gpt2c -g -lm -O3 -ffast-math # -march=native  # -fsanitize=address

