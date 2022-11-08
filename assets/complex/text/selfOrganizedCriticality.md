
## Intro
Self-organized criticality, is a property of dynamic systems which have a critical point as an attractor. In this type of model, we may set initial values rather arbitrarily, as the system itself, will evolve toward the critical state. At the criticality, the model displays spatial and/or temporal scale-invariance.

## Bak-Sneppen
The Bak-Sneppen model considers a random 1d chain, ``C = random(size)``, through time. At each timestep the smallest element and its two neighbors are replaced by new random numbers. This will obviously see that the values of C increase over time.

At some point when most values are large, it will be very likely that one of the newly replace numbers is the smallest. This then starts an avalance. These avalanches extend both through the spatial and temporal dimensions.