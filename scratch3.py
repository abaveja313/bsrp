from utils import ADHD200

a = ADHD200()
a.gen_data()

func_filenames = a.func

from nilearn.decomposition import CanICA

canica = CanICA(mask='mask_wmean_mrda0027000_session_1_rest_1.nii', n_components=20, smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=2,
                threshold=3., verbose=10, random_state=0)

canica.fit(func_filenames)

components_img = canica.masker_.inverse_transform(canica.components_)

print components_img

