#include "colors.inc"    

background { color Cyan }

camera {
  location <0, 2, -3>
  look_at <0, 1, 2>
}


box {
  <-1, -1, 1.5>, <1, 1, 2.5>
  texture {
    pigment { color Yellow }
  }
}


light_source { <2, 4, -3> color White}