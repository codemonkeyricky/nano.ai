all:
	gcc gpt2.c -o gpt2c -g -lm -O3 -ffast-math -march=native  # -fsanitize=address

