from data import create_particular_stimulus

# Fix Object 1 as (S, T), Generate Object 2 with (S, T')

# Then, create images with (S, T), (S'', T') for all possible S''
# Repeat for (S, T), (S'', T) for all possible S''

# Experiment: Average the 16 vectors in both conditions to create a T' vector and T vector.
# T' vector = mean(S) + T' / T vector = mean(S) + T
# See if you can subtract the T' vector and add the T vector from the residual stream of Object 2
# to change the model's decision from Different to Same.
# -mean(S) - T' + mean(S) + T  should remove T' and add T if Texture exists in a different linear subspace as Shape

# Repeat the same process for shape


ALL_SHAPES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
ALL_TEXTURES = [
    (32, 1),
    (32, 4),
    (32, 16),
    (32, 32),
    (96, 1),
    (96, 4),
    (96, 16),
    (96, 32),
    (160, 1),
    (160, 4),
    (160, 16),
    (160, 32),
    (224, 1),
    (224, 4),
    (224, 16),
    (224, 32),
]


def create_texture_linear_subspace_stimuli(
    shape,
    texture_1,
    texture_2,
    position_1,
    position_2,
    outdir,
    buffer_factor=8,
    im_size=224,
    patch_size=32,
):
    # Create base image. Shape is the same, texture is different
    create_particular_stimulus(
        shape,
        shape,
        texture_1,
        texture_2,
        position_1,
        position_2,
        outdir,
        "base",
        buffer_factor,
        im_size,
        patch_size,
    )

    create_particular_stimulus(
        shape,
        shape,
        texture_1,
        texture_1,
        position_1,
        position_2,
        outdir,
        "same",
        buffer_factor,
        im_size,
        patch_size,
    )

    # Create Texture 1 set
    for idx, shape_prime in enumerate(ALL_SHAPES):
        create_particular_stimulus(
            shape,
            shape_prime,
            texture_1,
            texture_1,  # Fix Texture 2 as same as Texture 1
            position_1,
            position_2,
            outdir,
            f"Texture_1_{idx}",
            buffer_factor,
            im_size,
            patch_size,
        )

    # Create Texture 2 set
    for idx, shape_prime in enumerate(ALL_SHAPES):
        create_particular_stimulus(
            shape,
            shape_prime,
            texture_1,
            texture_2,  # Fix Texture 2 as same as base
            position_1,
            position_2,
            outdir,
            f"Texture_2_{idx}",
            buffer_factor,
            im_size,
            patch_size,
        )


def create_shape_linear_subspace_stimuli(
    shape_1,
    shape_2,
    texture,
    position_1,
    position_2,
    outdir,
    buffer_factor=8,
    im_size=224,
    patch_size=32,
):
    # Create base image. Texture is the same, shape is different
    create_particular_stimulus(
        shape_1,
        shape_2,
        texture,
        texture,
        position_1,
        position_2,
        outdir,
        "base",
        buffer_factor,
        im_size,
        patch_size,
    )

    create_particular_stimulus(
        shape_1,
        shape_1,
        texture,
        texture,
        position_1,
        position_2,
        outdir,
        "same",
        buffer_factor,
        im_size,
        patch_size,
    )

    # Create Shape 1 set
    for idx, texture_prime in enumerate(ALL_TEXTURES):
        create_particular_stimulus(
            shape_1,
            shape_1,  # Fix shape 2 as the same as shape 1
            texture,
            texture_prime,  # Vary the texture
            position_1,
            position_2,
            outdir,
            f"Shape_1_{idx}",
            buffer_factor,
            im_size,
            patch_size,
        )

    # Create Shape 2 set
    for idx, texture_prime in enumerate(ALL_TEXTURES):
        create_particular_stimulus(
            shape_1,
            shape_2,  # Fix shape 2 as the same as base
            texture,
            texture_prime,  # Vary the texture
            position_1,
            position_2,
            outdir,
            f"Shape_2_{idx}",
            buffer_factor,
            im_size,
            patch_size,
        )


if __name__ == "__main__":
    create_texture_linear_subspace_stimuli(
        shape=0,
        texture_1=ALL_TEXTURES[2],
        texture_2=ALL_TEXTURES[4],
        position_1=(0, 0),
        position_2=(4, 4),
        outdir="stimuli/Interventions/Texture_Subspace_Set1",
    )

    create_shape_linear_subspace_stimuli(
        shape_1=0,
        shape_2=10,
        texture=ALL_TEXTURES[2],
        position_1=(0, 0),
        position_2=(4, 4),
        outdir="stimuli/Interventions/Shape_Subspace_Set1",
    )
