
zernike bead doesnt care about medium RI is this ok? simplification

make the optimization weights by which all of the parameters are multiplied an input parameter? this should probably not often be set but might be able to dial it in

add the zernike indices to the output. perhaps a new struct which could be used throughout could include the base polynomials, the mag indices and phase indices, then they dont need to be chucked around independently

create the tensorflow graph and analyze it and see if it can be rewritten in different language

phiz too large in new datastack maybe something to do with immersion ri etc in input parameters

make the 3d case more explicit in the roi selection
