#include <stdio.h>

#include "units/test_utils_functional.h"

static int test(const char * description, const int error);

int main(int argc, char * argv[]) {

    int error = 0;

    error += test("Testing relu................. ", test_utils_functional_relu());
    error += test("Testing sigmoid.............. ", test_utils_functional_sigmoid());

    return error;

}

static int test(const char * description, const int error) {

    printf("%s", description);
    if (error == 0) {
        printf("[\033[1;32mPASSED\033[0m]\n");
    }
    else {
        printf("[\033[1;31mFAILED with error code %d\033[0m]\n", error);
    }

    return error;

}


