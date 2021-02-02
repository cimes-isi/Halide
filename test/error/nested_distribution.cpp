#include "Halide.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    Func img("img");
    Func output("output");
    Var x, y;
    img(x, y) = x + y;
    output(x, y) = img(x, y);
    img.compute_at(output_img, x)
       .distribute(y); // no nested distribution
    output_img.distribute(y); 

    printf("Success!\n");
    return 0;
}
