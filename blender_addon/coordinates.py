"""Coordinate boundaries shared by native viewing and trainer export.

Dataset/preview: Blender Z-up, with only CAMERA-LOCAL OpenCV pose rebasing.
External PLY/SOG: existing SuperSplat convention, (x,y,z) -> (x,-z,y).
This is a viewer convention, not a universal COLMAP world-up assumption.
Positions, Gaussian orientations and directional SH must change together.
"""
import math

BLENDER_TO_EXTERNAL = ((1.,0.,0.),(0.,0.,-1.),(0.,1.,0.))
EXTERNAL_TO_BLENDER = ((1.,0.,0.),(0.,0.,1.),(0.,-1.,0.))


def rotate_pose(positions, quaternions, rotation, np):
    matrix = np.asarray(rotation, dtype=np.float64)
    # Our file conventions differ by quarter turns about X only.
    angle = math.atan2(matrix[2,1], matrix[1,1])
    c, s = math.cos(angle/2), math.sin(angle/2)
    q = np.asarray(quaternions,dtype=np.float32)
    q = q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)
    w,x,y,z = q.T
    result = np.stack((c*w-s*x,c*x+s*w,c*y-s*z,c*z+s*y),axis=1)
    return (positions @ matrix.T).astype(np.float32), result.astype(np.float32)


_SH_C1 = 0.4886025119029199
_SH_C2 = (
    1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
    -1.0925484305920792, 0.5462742152960396,
)
_SH_C3 = (
    -0.5900435899266435, 2.890611442640554, -0.4570457994644658,
    0.3731763325901154, -0.4570457994644658, 1.445305721320277,
    -0.5900435899266435,
)

#: ``f_rest`` column ranges per band, in the order the 3DGS PLY stores them.
#: Band 0 - the DC term in ``f_dc`` - is invariant under rotation.
_SH_BANDS = ((0, 3), (3, 5), (8, 7))


def _sh_rest_basis(directions, np):
    """The 15 non-DC basis values per unit direction, exactly as the
    reference 3DGS rasterizer evaluates them."""
    x = directions[:, 0]
    y = directions[:, 1]
    z = directions[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    return np.stack(
        (
            -_SH_C1 * y,
            _SH_C1 * z,
            -_SH_C1 * x,
            _SH_C2[0] * xy,
            _SH_C2[1] * yz,
            _SH_C2[2] * (2.0 * zz - xx - yy),
            _SH_C2[3] * xz,
            _SH_C2[4] * (xx - yy),
            _SH_C3[0] * y * (3.0 * xx - yy),
            _SH_C3[1] * xy * z,
            _SH_C3[2] * y * (4.0 * zz - xx - yy),
            _SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            _SH_C3[4] * x * (4.0 * zz - xx - yy),
            _SH_C3[5] * z * (xx - yy),
            _SH_C3[6] * x * (xx - 3.0 * yy),
        ),
        axis=1,
    )


_SH_ROTATION_CACHE = {}


def sh_rest_rotation(rotation, np):
    """Per-band coefficient matrices for a world rotation, or None.

    Each SH band is closed under rotation, so the matrix is recovered by
    evaluating the basis on well-spread directions and solving the small
    linear system - exact to floating-point, and far less error-prone than
    hand-derived Wigner matrices. The result is checked for orthogonality
    before it is used; anything else means the derivation is unsound for
    this rotation and the bands are left alone rather than corrupted.
    """
    key = tuple(tuple(float(value) for value in row) for row in rotation)
    if key in _SH_ROTATION_CACHE:
        return _SH_ROTATION_CACHE[key]
    matrix = np.asarray(key, dtype=np.float64)
    count = 4096
    index = np.arange(count, dtype=np.float64) + 0.5
    cos_theta = 1.0 - 2.0 * index / count
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    phi = index * math.pi * (1.0 + math.sqrt(5.0))
    directions = np.stack(
        (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta), axis=1
    )
    # ``directions @ matrix`` is R^-1 d for a rotation, which is the
    # direction the rotated color function has to be sampled at.
    here = _sh_rest_basis(directions, np)
    there = _sh_rest_basis(directions @ matrix, np)
    blocks = []
    for start, width in _SH_BANDS:
        columns = slice(start, start + width)
        block = np.linalg.lstsq(here[:, columns], there[:, columns], rcond=None)[0]
        residual = float(np.abs(here[:, columns] @ block - there[:, columns]).max())
        drift = float(
            np.abs(block @ block.T - np.eye(width)).max()
        )
        if not (residual < 1.0e-6 and drift < 1.0e-6):
            _SH_ROTATION_CACHE[key] = None
            return None
        blocks.append(block)
    _SH_ROTATION_CACHE[key] = tuple(blocks)
    return _SH_ROTATION_CACHE[key]


def rotate_sh_rest(sh_rest, blocks, np):
    """Rotate an ``[N, coefficients, 3]`` block of non-DC SH bands."""
    if not blocks or sh_rest.size == 0:
        return sh_rest
    rotated = np.array(sh_rest, dtype=np.float32, copy=True)
    available = int(sh_rest.shape[1])
    for (start, width), block in zip(_SH_BANDS, blocks):
        if start + width > available:
            break
        band = rotated[:, start:start + width, :]
        rotated[:, start:start + width, :] = np.einsum(
            "mn,inc->imc", block, band, optimize=True
        ).astype(np.float32, copy=False)
    return rotated



def to_blender(arrays, source_space, np):
    if source_space not in {'EXPORTED_OPENCV','LEGACY_Y_UP'}:
        return arrays
    positions, scales, quaternions, opacity, sh = arrays
    rotation = EXTERNAL_TO_BLENDER if source_space == 'EXPORTED_OPENCV' else BLENDER_TO_EXTERNAL
    positions, quaternions = rotate_pose(positions,quaternions,rotation,np)
    # Very old Y-up exports rotated the pose but left SH in Blender space.
    if source_space == 'EXPORTED_OPENCV' and sh.shape[1] > 1:
        sh = sh.copy()
        sh[:,1:,:] = rotate_sh_rest(sh[:,1:,:],sh_rest_rotation(rotation,np),np)
    return positions,scales,quaternions,opacity,sh
