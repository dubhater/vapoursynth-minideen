Description
===========

MiniDeen is a spatial denoising filter. It replaces every pixel with
the average of its neighbourhood.

This is a port of the "a2d" method from the Avisynth plugin Deen,
version beta 2.


Usage
=====
::

    minideen.MiniDeen(clip clip[, int[] radius=1, int[] threshold=10, int[] planes=all])


Parameters:
    *clip*
        A clip to process. It must have constant format and it must be
        8..16 bit with integer samples.

    *radius*
        Size of the neighbourhood. Must be between 1 (3x3) and 7
        (15x15).

        Default: 1 for the first plane, and the previous plane's radius
        for the other planes.

    *threshold*
        Only pixels that differ from the center pixel by less than the
        *threshold* will be included in the average. Must be between 2
        and 255.

        The threshold is scaled internally according to the bit depth.

        Smaller values will filter more conservatively.

        Default: 10 for the first plane, and the previous plane's
        threshold for the other planes.

    *planes*
        Planes to filter. Planes that aren't filtered will be copied
        from the input.

        Default: all.


Compilation
===========

::

    mkdir build && cd build
    meson ../
    ninja


License
=======

ISC.
