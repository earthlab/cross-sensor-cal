import math
import numpy as np
import itertools as it
from functools import partial
from multiprocessing.dummy import Pool


class MesmaCore:
    """
    Multiple Endmember Signal Mixture Analysis: calculate SMA for multiple endmember combinations and select the
    best fit based on the lowest RMSE.

    Citations:
    MESMA: Roberts, D.A., Gardner, M., Church, R., Ustin, S., Scheer, G., Green, R.O., 1998, Mapping Chaparral in the
    Santa Monica Mountains using Multiple Endmember Spectral Mixture Models, Remote Sensing of Environment, 65,
    p. 267-279.

    Multilevel fusion: Roberts, D.A., Dennison, P.E., Gardner, M., Hetzel, Y., Ustin, S.L., Lee, C., 2003, Evaluation of
    the Potential of Hyperion for Fire Danger Assessment by Comparison to the Airborne Visible/Infrared Imaging
    Spectrometer, IEEE Transactions on Geoscience and Remote Sensing, 41, p. 1297-1310.

    Spectral band weighing: Somers, B., Delalieux, S, Stuckens, J , Verstraeten, W.W, Coppin, P., 2009, A weighted
    linear spectral mixture analysis approach to address endmember variability in agricultural production systems,
    International Journal of Remote Sensing, 30, p. 139-147.

    Spectral band selection: Somers, B., Delalieux, S., Verstraeten, W.W., van Aardt, J.A.N., Albrigo, G., Coppin, P.,
    2010, An automated waveband selection technique for optimized hyperspectral mixture analysis. International
    Journal of Remote Sensing, 31, p. 5549-5568.
    """

    def __init__(self, n_cores: int = 1):
        """
        :param n_cores: the numbers of cores you are dedicating to this process
        """
        # image variables
        self.n_pixels = None  # total number of pixels
        self.n_bands = None  # number of bands
        self.image_as_line = None  # image rearranged as a line

        # library variable
        self.library = None
        self.n_classes = None  # number of different classes
        self.look_up_table = None  # dict with all models (library indices): org. per level & class model
        self.em_per_class = None  # dict with for each endmember class, a list of all library indices

        # constraints ('-9999' when not used): min max fraction, min max shade, max rmse, residual threshold, # bands
        self.constraints = None

        # variables for unstable zone unmixing
        self.class_mean = None  # mean spectrum per endmember class
        self.class_std = None  # standard deviation per endmember class
        self.use_band_weighing = False
        self.use_band_selection = False
        self.bands_selection_values = (0.99, 0.01)

        # multiprocessing
        self.pool = Pool(int(n_cores))

        # log
        self.log = print

    def _set_image(self, image: np.array, no_data_pixels: np.array):
        """ Store the image as a line (2D array).

        :param image: first dimension are bands, scaled to reflectance (between 0 and 1), without bad bands
        :param no_data_pixels: 1D indices of pixels that contain no data
        """
        image = np.float32(image)
        if image.ndim == 2:
            self.n_bands = image.shape[0]
            self.n_pixels = image.shape[1] - len(no_data_pixels)
            self.image_as_line = np.delete(image, no_data_pixels, axis=1)
        else:
            self.n_bands = image.shape[0]
            n_pixels = image.shape[1] * image.shape[2]
            self.image_as_line = np.reshape(image, [self.n_bands, n_pixels])
            self.image_as_line = np.delete(self.image_as_line, no_data_pixels, axis=1)
            self.n_pixels = n_pixels - len(no_data_pixels)

    def _set_library(self, library: np.array):
        """ Store the library as a float 32 object

        :param library: spectra as rows, scaled to reflectance (between 0 and 1), without bad bands
        """
        self.library = np.float32(library)

    def _subtract_shade(self, shade_spectrum: np.array):
        """ Correct image and library for the photometric shade.

        :param shade_spectrum: single spectrum of photometric shade
        """
        self.image_as_line = self.image_as_line - shade_spectrum
        self.library = self.library - shade_spectrum

    def _sma(self, look_up_table: np.array, bands: np.array):
        """ SMA (Signal Mixture Analysis): calculate the fraction of each endmember, based on SVD.

        :param look_up_table: all endmember combinations (=models) for a given complexity level and class-model
        :param bands: the bands used for unmixing
        :return: for each pixel the best model and the fractions and rmse belonging to that model
        """
        # get the endmembers from the library
        endmembers = self.library[:, look_up_table.T][bands, :, :]

        # run a singular value decomposition on the endmember array where:
        #   w = vector (n_em) with the eigenvalues
        #   u = orthogonal array (n_b x n_b) used in decomposition
        #   v = orthogonal array (n_em x n_em) used in decomposition
        #  --> then   library = u . diagonal(w) . transpose(v)
        u, w, v = np.linalg.svd(endmembers.T, full_matrices=False)

        # inverse endmembers
        temp = v * 1 / w[:, :, np.newaxis]
        endmembers_inverse = np.array([np.dot(y, x) for (x, y) in zip(temp, u)])

        # fractions
        fractions = np.einsum('ijk, kl -> ijl', endmembers_inverse, self.image_as_line[bands, :])

        # residuals + root mean square error
        residuals = self.image_as_line[bands, np.newaxis, :] - np.einsum('ijk, kjl -> ikl', endmembers, fractions)
        rmse = np.sqrt(np.sum(residuals * residuals, axis=0) / len(bands))

        # shade fraction (last column of the fractions array)
        shade = 1 - np.sum(fractions, axis=1)
        fractions = np.concatenate((fractions, shade[:, np.newaxis, :]), axis=1)

        # check the user constraints and set rmse to 9999 where they are not met
        rmse = self._sma_constraints(rmse, fractions, residuals)

        # select the best model based on the lowest rmse
        index = np.argmin(rmse, axis=0)
        best_rmse = rmse[index, np.arange(self.n_pixels)]
        best_model = look_up_table[index]
        best_fraction = fractions[index, :, np.arange(self.n_pixels)]

        return best_model, best_fraction, best_rmse

    def _sma_weighted(self, look_up_table: np.array, bands: np.array, class_model: np.array):
        """ SMA (Signal Mixture Analysis): calculate - per pixel - the fraction of each endmember, based on SVD.

        :param look_up_table: all endmember combinations (=models) for a given complexity level and class-model
        :param bands: the bands used for unmixing
        :param class_model: class model (e.g. GV-SOIL)
        :return: for each pixel the best model and the fractions and rmse belonging to that model
        """
        weights = self._weights(class_model, bands)
        pixels = self.image_as_line[bands, :] * weights

        rmse = np.zeros((self.n_pixels, len(look_up_table)), dtype=np.float32)
        fractions = np.zeros((self.n_pixels, len(class_model) + 1, len(look_up_table)), dtype=np.float32)
        residuals = None

        for m, model in enumerate(look_up_table):
            # get the endmembers from the library and weigh them
            endmembers = self.library[:, model][bands, :, np.newaxis] * weights[:, np.newaxis, :]

            # run a singular value decomposition on the endmember array where:
            #   w = vector (n_em) with the eigenvalues
            #   u = orthogonal array (n_b x n_b) used in decomposition
            #   v = orthogonal array (n_em x n_em) used in decomposition
            #  --> then   library = u . diagonal(w) . transpose(v)
            u, w, v = np.linalg.svd(endmembers.T, full_matrices=False)

            # inverse endmembers + fraction for each endmember
            temp = v * 1 / w[:, :, np.newaxis]
            endmembers_inverse = np.array([np.dot(y, x) for (x, y) in zip(temp, u)])
            fractions[:, 0:-1, m] = np.einsum('ijk, ki -> ji', endmembers_inverse, pixels).T

            # residuals + root mean square error
            residuals = pixels - np.einsum('ijk, jk -> ik', endmembers, fractions[:, 0:-1, m].T)
            rmse[:, m] = np.sqrt(np.sum(residuals * residuals, axis=0) / len(bands))

        # shade fraction (last column of the fractions array)
        fractions[:, -1, :] = 1 - np.sum(fractions, axis=1)

        # check the user constraints
        rmse = self._sma_constraints(rmse, fractions, residuals)

        # select the best model based on the lowest rmse
        index = np.argmin(rmse, axis=1)
        best_rmse = rmse[np.arange(self.n_pixels), index]
        best_model = look_up_table[index, :]
        best_fraction = fractions[np.arange(self.n_pixels), :, index]

        return best_model, best_fraction, best_rmse

    def _sma_constraints(self, rmse: np.array, fractions: np.array, residuals: np.array) -> np.array:
        """ Apply the constraints on fractions, rmse and possibly residuals and set RMSE = 9999 where breached

        :param rmse: rmse from sma algorithm
        :param fractions: fractions from sma algorithm
        :param residuals: residuals from sma algorithm
        :return: same rmse array, but set to 9999 where constraints were breached
        """

        # when rmse is nan: return 9999
        rmse[np.isnan(rmse)] = 9999

        bad_models = np.zeros(rmse.shape, dtype=bool)

        if self.constraints[0] != -9999:
            fractions_min = np.amin(fractions[:, 0:-1, :], axis=1)
            bad_models[np.where(fractions_min < self.constraints[0])] = 1

        if self.constraints[1] != -9999:
            fractions_max = np.amax(fractions[:, 0:-1, :], axis=1)
            bad_models[np.where(fractions_max > self.constraints[1])] = 1

        if self.constraints[2] != -9999:
            bad_models[np.where(fractions[:, -1, :] < self.constraints[2])] = 1

        if self.constraints[3] != -9999:
            bad_models[np.where(fractions[:, -1, :] > self.constraints[3])] = 1

        if self.constraints[4] != -9999:
            bad_models[np.where(rmse > self.constraints[4])] = 1

        # apply the consecutive residuals constraint
        if self.constraints[5] != -9999:
            good_bands = np.abs(residuals) < self.constraints[5]
            self.constraints[6] = int(self.constraints[6])
            for band in np.arange(self.n_bands - self.constraints[6] + 1):
                bad_models[np.where(np.sum(good_bands[band:band + self.constraints[6], :, :], axis=0) == 0)] = 1

        rmse[bad_models] = 9999

        return rmse

    def _mesma_per_model(self, class_model: np.array, level: int):
        """ Run SMA for all models in a given  classes (e.g. GV-WATER) of a given levels (e.g. 2-EM, 3-EM)

        :param class_model: class model (e.g. GV-SOIL)
        :param level: the complexity level
        :return: for each pixel the best model and the fractions and rmse belonging to that model
        """
        if self.use_band_selection and level > 2:
            bands = self._szu(class_model)
        else:
            bands = np.arange(self.n_bands)

        # get the look-up table
        look_up_table = self.look_up_table[level][class_model]

        # decide on block number and size
        block_size = math.ceil(len(look_up_table) / (len(bands) * look_up_table.size * 4 / 50000))
        n_blocks = math.ceil(len(look_up_table) / block_size)

        # divide the look_up_table in blocks
        lut_blocks = np.empty(n_blocks, dtype=object)
        for b in range(n_blocks):
            start = b * block_size
            end = min((b + 1) * block_size, len(look_up_table))
            lut_blocks[b] = look_up_table[start:end, :]

        # pool
        # run sma per block
        if self.use_band_weighing and level > 2:
            pool_output = self.pool.map(partial(self._sma_weighted, bands=bands, class_model=class_model), lut_blocks)
        else:
            pool_output = self.pool.map(partial(self._sma, bands=bands), lut_blocks)

        # place each model output on the correct place
        models = np.zeros((self.n_pixels, len(class_model), n_blocks), dtype=np.int32) - 1
        fractions = np.zeros((self.n_pixels, len(class_model) + 1, n_blocks), dtype=np.float32)
        rmse = np.zeros((self.n_pixels, n_blocks), dtype=np.float32)
        for i, part in enumerate(pool_output):
            models[:, :, i] = part[0]
            fractions[:, :, i] = part[1]
            rmse[:, i] = part[2]

        # find the best model within each complexity level
        index = np.argmin(rmse, axis=1)
        best_rmse = rmse[np.arange(self.n_pixels), index]
        best_model = models[np.arange(self.n_pixels), :, index]
        best_fractions = fractions[np.arange(self.n_pixels), :, index]
        return best_model, best_fractions, best_rmse

    def _mesma_per_level(self, level: int):
        """ Run SMA for different model classes (e.g. GV-WATER) of a given levels (e.g. 2-EM, 3-EM, ... combinations)

        :param: level: the complexity level
        :return: for each pixel the best model and the fractions and rmse belonging to that model
        """
        if (self.use_band_selection or self.use_band_weighing) and level == 2:
            self.log("Standard MESMA is used for 2-EM models!")

        class_models = self.look_up_table[level].keys()

        models = np.zeros((self.n_pixels, self.n_classes, len(class_models)), dtype=np.int32) - 1
        fractions = np.zeros((self.n_pixels, self.n_classes + 1, len(class_models)), dtype=np.float32)
        rmse = np.zeros((self.n_pixels, len(class_models)), dtype=np.float32)

        for m, model in enumerate(class_models):
            m_indices = np.array(model)
            f_indices = np.append(m_indices, self.n_classes)  # append shade as last index
            models[:, m_indices, m], fractions[:, f_indices, m], rmse[:, m] = self._mesma_per_model(model, level)

        # find the best model within each complexity level
        index = np.argmin(rmse, axis=1)
        best_rmse = rmse[np.arange(self.n_pixels), index]
        best_model = models[np.arange(self.n_pixels), :, index]
        best_fractions = fractions[np.arange(self.n_pixels), :, index]
        return best_model, best_fractions, best_rmse

    def _mesma(self, fusion_value: float = 0.007):
        """ Run MESMA for different levels (e.g. 2-EM, 3-EM, ... combinations)

        :param fusion_value: only select a model of higher complexity (e.g. 3-EM over 2-EM) if the RMSE is better
                                   with at least this value
        :return: for each pixel the best model and the fractions and rmse belonging to that model
        """
        levels = self.look_up_table.keys()
        n_levels = len(levels)

        # run MESMA per complexity level
        rmse = np.zeros((self.n_pixels, n_levels), dtype=np.float32)
        models = np.zeros((self.n_pixels, self.n_classes, n_levels), dtype=np.int32)
        fractions = np.zeros((self.n_pixels, self.n_classes + 1, n_levels), dtype=np.float32)

        self.log("processing ", end='')
        for l, level in enumerate(levels):
            self.log("{}-EM models..".format(level), end='')
            models[:, :, l], fractions[:, :, l], rmse[:, l] = self._mesma_per_level(level)

        # reset RMSE if it is not at least *complexity_threshold* better than the previous complexity level
        difference = rmse[:, :-1] - rmse[:, 1:]
        remove_ind = difference < fusion_value
        remove_ind = np.insert(remove_ind, 0, np.zeros(self.n_pixels), axis=1)
        rmse[remove_ind] = 9999

        # find the best model over all complexity levels
        index = np.argmin(rmse, axis=1)
        best_rmse = rmse[np.arange(self.n_pixels), index]
        best_model = models[np.arange(self.n_pixels), :, index]
        best_fractions = fractions[np.arange(self.n_pixels), :, index]

        # reset values to not modeled where final RMSE is still 9999
        indices = np.where(best_rmse == 9999)
        best_model[indices, :] = -1
        best_fractions[indices, :] = 0
        return best_model, best_fractions, best_rmse

    def _isi_preparation(self):
        """ Preparation for the ISI (Instability Index): mean and standard deviation spectrum of each class. """

        self.class_mean = np.zeros((self.n_classes, self.n_bands), dtype=np.float32)
        self.class_std = np.zeros((self.n_classes, self.n_bands), dtype=np.float32)
        for i, key in enumerate(self.em_per_class):
            self.class_mean[i, :] = np.mean(self.library[:, self.em_per_class[key]], axis=1)
            if self.library[:, self.em_per_class[key]].shape[1] != 1:
                self.class_std[i, :] = np.std(self.library[:, self.em_per_class[key]], axis=1, ddof=1)
            else:
                self.class_std[i, :] = np.std(self.library[:, self.em_per_class[key]], axis=1)

    def _isi(self, class_model: np.array) -> np.array:
        """
        :param class_model: class model (e.g. GV-SOIL). Requires at least two classes.
        :return: ISI (Instability Index)
        """
        combos = list(it.combinations(class_model, 2))

        # Equation 6 from Somers et al, 2009, p. 142
        divisor = len(combos)
        denominator = np.zeros(self.n_bands)
        for (one, two) in combos:
            denominator += (self.class_std[one, :] + self.class_std[two, :]) / \
                           np.abs(self.class_mean[one, :] - self.class_mean[two, :])

        return denominator / divisor

    def _weights(self, class_model: np.array, bands: np.array) -> np.array:
        """
        :param class_model: class model (e.g. GV-SOIL). Requires at least two classes.
        :param bands: the bands used for unmixing
        :return: weights for the spectral band weighing. Somers et al (2009), p141-142, equation 3, 4 and 7
        """
        a = 1 / self.image_as_line[bands, :] * np.max(self.image_as_line[bands, :], axis=0)
        # a = max(pixel) / pixel
        b = 1 / self._isi(class_model)[bands]
        return (a.T * b).T

    def _szu(self, class_model: np.array) -> np.array:
        """
        :param class_model: class model (e.g. GV-SOIL). Requires at least two classes.
        :return: selected bands for a given class model. Somers et al (2010), p5553-5556
        """
        # calculate SI = 1/ISI
        si = 1 / self._isi(class_model)

        # iterative band selection
        used_bands = []
        count = 0
        threshold = self.bands_selection_values[0]

        while sum(si) > (len(si) * -1):
            # get the index of the max SI
            si_max_ind = np.argmax(si)
            si[si_max_ind] = -1  # avoid further usage
            used_bands.append(si_max_ind)

            # correlation
            corr = np.ones(len(si), dtype=np.float32) + threshold
            for band in np.where(si != -1)[0]:
                corr[band] = np.corrcoef(self.library[si_max_ind, :], self.library[band, :])[1, 0]

            si[np.where(corr > threshold)] = -1

            # adapt threshold
            threshold = threshold - self.bands_selection_values[1] * (2 ** count)
            count += 1

        return np.sort(used_bands)

    def _residual_image(self, models: np.array, fractions: np.array) -> np.array:
        """
        :param models: the models as a result of the MESMA algorithm
        :param fractions: the fractions as a result of the MESMA algorithm
        :return: the residuals image, calculated based on the selected models and their fractions
        """
        endmembers = self.library[:, models]
        residuals = self.image_as_line - np.sum(fractions[np.newaxis, :, 0:-1] * endmembers, axis=2)
        unmodeled = np.where(np.sum(models, axis=1) == -self.n_classes)
        residuals[:, unmodeled] = 0
        return residuals

    def execute(self, image: np.array, library: np.array, look_up_table: dict, em_per_class: dict,
                constraints: list = (-0.05, 1.05, 0., 0.8, 0.025, -9999, -9999), fusion_value: float = 0.007,
                no_data_pixels: tuple = (), shade_spectrum: np.array = None, residual_image: bool = False,
                use_band_weighing: bool = False, use_band_selection: bool = False,
                bands_selection_values: tuple = (0.99, 0.01), log: callable = print) -> tuple:
        """
        Execute MESMA. Process input and output.

        In case band weighing or band selection algorithms are used, no residual image or residual constraints can be
        used.

        Returns 3 images [a pixels * b pixels * c bands]:

            * the best model [nb of bands = nb of classes] - each band contains the library spectra number per class
            * the model's fractions [nb of bands = nb of classes + 1], including a shade fraction
            * the model's RMSE [nb of bands = 1]
            * [optional] a residual image

        Value of unmodeled pixels in output:

            * models: -1
            * fractions: 0
            * rmse: 9999
            * residual_image: 0

        Value of pixels with no data in output:

            * models: -2
            * fractions: 0
            * rmse: 9998
            * residual_image: 0

        :param image: image, scaled to reflectance, without bad bands
        :param library: spectral library with spectra as columns, scaled to reflectance, without bad bands
        :param look_up_table: all endmember combinations (=models) for MESMA; ordered per complexity level and per
                              class-model; n_models x n_endmembers
        :param em_per_class: a list of all library indices per endmember class
        :param constraints: min + max endmember fraction, min + max shade fraction, max rmse,
                            residual reflectance threshold + max number of consecutive bands exceeding threshold.
                            set value to -9999 if not used.
        :param no_data_pixels: indices of pixels that contain no data (result of np.where)
        :param shade_spectrum: single spectrum of photometric shade
        :param fusion_value: only select a model of higher complexity (e.g. 3-EM over 2-EM) of the RMSE is better
                                   with at least this value
        :param residual_image: output the residuals as an image (ignored when using band weighing or -selection)
        :param use_band_weighing: use the weighted linear spectral mixture analysis (Somers et al, 2009)
        :param use_band_selection: use the bands selection algorithm (Somers et al, 2010)
        :param bands_selection_values: correlation threshold and decrease for the band selection algorithm
        :param log: log function
        :return: images with the best model for each pixel, the model fractions and rmse belonging {+ evt. residuals)
        """

        self.log = log

        # reference to gui progress bar (if exists)
        if np.nanmax(image) > 1:
            raise ValueError('The algorithm detected image values larger than 1: '
                             'reflectance scale factor not set correctly.')

        print(library)
        if np.nanmax(library) > 1:
            raise ValueError('The algorithm detected library values larger than 1:'
                             'reflectance scale factor not set correctly.')

        # set up environment
        image_dimensions = image.shape[1:]
        n_pixels = np.prod(image_dimensions)
        self.n_classes = len(em_per_class)
        # no_data_pixels
        if no_data_pixels:
            no_data_pixels = np.ravel_multi_index(no_data_pixels, image_dimensions)
        else:
            no_data_pixels = np.array([], dtype=int)
        zero_pixels = np.ravel_multi_index(np.where(np.max(image, axis=0) == 0), image_dimensions)
        no_data_pixels = np.unique(np.concatenate((no_data_pixels, zero_pixels)))
        # reverse of no_data_pixels
        data_pixels = np.ones(n_pixels, dtype=bool)
        data_pixels[no_data_pixels] = 0

        residuals = None

        models_expanded = np.ones((n_pixels, self.n_classes), dtype=int) * -2
        fractions_expanded = np.ones((n_pixels, self.n_classes + 1), dtype=float) * 0
        rmse_expanded = np.ones(n_pixels, dtype=float) * 9998
        if residual_image:
            residuals_expanded = np.zeros((image.shape[0], n_pixels), dtype=float)
        else:
            residuals_expanded = None

        if len(no_data_pixels) != n_pixels:
            self._set_image(image, no_data_pixels)
            self._set_library(library)
            self.look_up_table = look_up_table
            self.em_per_class = em_per_class
            self.constraints = list(constraints)
            if shade_spectrum is not None:
                self._subtract_shade(shade_spectrum)
            self.use_band_weighing = use_band_weighing
            self.use_band_selection = use_band_selection
            self.bands_selection_values = bands_selection_values
            if use_band_weighing or use_band_selection:
                self._isi_preparation()  # prepare often used values
                residual_image = False  # not possible so ignored
                self.constraints[5] = -9999  # not possible so ignored
                self.constraints[6] = -9999  # not possible so ignored

            # run MESMA
            models, fractions, rmse = self._mesma(fusion_value=fusion_value)
            if residual_image:
                residuals = self._residual_image(models, fractions)

            # return variables in the original form of the image
            models_expanded[data_pixels, :] = models
            fractions_expanded[data_pixels, :] = fractions
            rmse_expanded[data_pixels] = rmse
            if residual_image:
                residuals_expanded[:, data_pixels] = residuals

        if residual_image:
            return (np.reshape(models_expanded.T, (self.n_classes,) + image_dimensions),
                    np.reshape(fractions_expanded.T, (self.n_classes + 1,) + image_dimensions),
                    np.reshape(rmse_expanded, image_dimensions),
                    np.reshape(residuals, image.shape))

        else:
            return (np.reshape(models_expanded.T, (self.n_classes,) + image_dimensions),
                    np.reshape(fractions_expanded.T, (self.n_classes + 1,) + image_dimensions),
                    np.reshape(rmse_expanded, image_dimensions),
                    None)


class MesmaModels:
    """
    Create the MESMA look-up-table from a list of classes and user input. No GUI/CLI.

    DEFINITIONS:

        * endmember: spectrum or signal from a Spectral Library = 'EM'
        * class: logical group of endmembers, e.g. 'green vegetation' or 'soil'
        * endmember-model: combination of endmembers used for unmixing
        * class-model: endmember-models grouped by class-level, e.g. all 'green vegetation-soil' models
        * level: class-models grouped by the number of classes (e.g. all 3-EM models)
    """

    def __init__(self):
        # variables on the classes
        self.unique_classes = None  # all unique classes
        self.n_classes = None  # number of different classes

        self.em_per_class = dict()  # indices of all endmembers, grouped per class
        self.n_em_per_class = None  # list of the number of endmembers per class

        # variables on the levels
        self.level_yn = None  # yes-no list of levels that are part of the selection
        self.class_per_level = dict()  # dict of yes-no lists of classes selected per level

        # variables on the class-models
        self.class_models = dict()  # dict of all class-models per level
        self.class_models_yn = dict()  # dict of yes-no lists of all class-models selected per level

    def setup(self, class_list: np.array):
        """
        Default set up of the model selection: select all 2-EM and 3-EM models.

        :param class_list: [array-of-strings] A class for each endmember in the library.
        """
        print(class_list)
        class_list = np.asarray([str(x).lower() for x in class_list])  # set all in lowercase
        print(class_list)
        self.unique_classes = np.unique(class_list)
        print(self.unique_classes)
        self.n_classes = len(self.unique_classes)
        self.n_em_per_class = np.zeros(self.n_classes, dtype=int)

        # per class: save all the indices from the library
        for i in np.arange(self.n_classes):
            indices = np.where(class_list == self.unique_classes[i])[0]
            self.em_per_class[i] = indices
            self.n_em_per_class[i] = len(indices)

        # initialise x_em_model_yn to zero
        self.level_yn = np.zeros(self.n_classes + 2, dtype=bool)
        print(self.level_yn, self.n_classes + 2)
        # initialise class_per_x_em_model to zeros
        for level in np.arange(2, self.n_classes + 2):
            self.class_per_level[level] = np.zeros(self.n_classes, dtype=bool)

        # initially select max 3EM models (2 classes + shade) in which select all classes
        self.select_level(state=True, level=2)
        self.select_level(state=True, level=3)
        for i in np.arange(self.n_classes):
            self.select_class(state=True, index=i, level=2)
            self.select_class(state=True, index=i, level=3)

        # initially we pick all the classes for each complexity level so nothing more to do here

    def select_level(self, state: bool, level: int):
        """
        Add/remove a level from the selection.

            * Selecting a level for the first time does not automatically select any classes/class-models in that level.
            * Deselecting a level leaves all settings of class/class-model selections intact.
            * Selecting a level for the second time (or more) re-instated the previous settings of that level.

        :param state: True = select, False = deselect.
        :param level: The complexity level the user wants to select.
        """
        self.level_yn[level] = state

    def select_class(self, state: bool, index: int, level: int):
        """
        Add/remove a class of a given level. Automatically selects all class-models of all selected classes. For
        level 3 (= 3-EM models), at least 2 classes must be selected, etc.

        :param state: True = select, False = deselect.
        :param index: The index of the class, based on the list of unique lowercase classes.
        :param level: The complexity level in which the user wants to select.
        """
        self.class_per_level[level][index] = state

        # recalculate all class-models with the new subset of classes, if enough are selected
        used_classes = np.where(self.class_per_level[level])[0]
        self.class_models[level] = list(it.combinations(used_classes, level - 1))

        # reset the selection of these individual models to 1
        self.class_models_yn[level] = np.ones(len(self.class_models[level]), dtype=bool)

    def select_model(self, state: bool, index: int, level: int):
        """
        Add/remove an individual class-model.

        :param state: True = select, False = deselect.
        :param index: The index of the class-model
        :param level: The complexity level in which the user wants to select.
        """
        self.class_models_yn[level][index] = state

    def total(self) -> int:
        """
        Get the total number of models in the current selection.

        :return: The total number of models in the current selection.
        """
        total = 0
        levels = np.where(self.level_yn)[0]
        for level in levels:
            total += self.total_per_level(level)

        return total

    def total_per_level(self, level: int) -> int:
        """
        Get the total number of models of a given level in the current selection.

        :param level: The complexity level.
        :return: The total number of models of a given level in the current selection.
        """

        if level not in self.class_models.keys():
            return 0

        total = 0
        for (model, yn) in zip(self.class_models[level], self.class_models_yn[level]):
            if yn:
                total += self.total_per_class_model(model)
        return total

    def total_per_class_model(self, model: tuple) -> int:
        """
        Get the total number of models of a given class-model in the current selection.

        :param model: The class-model.
        :return: The total number of models of a given class-model in the current selection.
        """
        return int(np.prod(self.n_em_per_class[np.array(model)]))

    def max_digits(self) -> int:
        """
        Get the maximum number of digits for the GUI, in order to be able to display the number of models.

        :return: The maximum number of digits for the GUI, in order to be able to display the number of models.
        """
        max_total = np.prod(np.array(self.n_em_per_class[np.arange(self.n_classes)], dtype=np.uint64)) * \
            self.n_classes * self.n_classes
        n_digits = len(str(max_total))
        n_spaces = int(n_digits / 3)  # add one digit per white space between thousands
        return n_digits + n_spaces

    def return_look_up_table(self) -> dict:
        """
        Get the actual look-up-table as a dictionary with a key-value pair for each level. Each level's value is another
        dictionary with a key-value pair for each class-model. Each class-model has a numpy array of all the
        endmember-combinations for that class-model.

        This look-up-table is a required input for MESMA.

        :return: The look-up-table as a dictionary [levels] of dictionaries [class-models] of numpy-arrays [models].
        """
        look_up_table = dict()
        selected_levels = np.where(self.level_yn)[0]
        for level in selected_levels:

            if len(self.class_models[level]) != 0:
                look_up_table[level] = dict()

                for c, class_model in enumerate(self.class_models[level]):

                    if self.class_models_yn[level][c]:
                        # we don't know beforehand how many arguments (e.g. 3 in GV-NPV-WATER or 2 in NPV-SOIL)
                        # we will need so we append the arguments and unpack them later
                        args = []  # list of arguments
                        for cl in class_model:
                            args.append(self.em_per_class[cl])

                        # we use the itertools.product function to generate all EM combinations of a model (e.g. GV-NPV)
                        look_up_table[level][class_model] = np.array(list(it.product(*args)))

        return look_up_table

    def summary(self) -> str:
        """
        Get a summary of the selected models. This is designed specifically to display in the GUI application.

        :return: A summary of the selected models, specifically to display in a GUI application [multi-line-str].
        """
        summary = 'You selected: '

        levels = np.where(self.level_yn)[0]

        for level in levels:
            summary += '{}-EM Models'.format(level)
            if level != levels[-1]:
                summary += ', '
            else:
                summary += '\n'

        summary += 'Total number of models: {} \n\n'.format(self.total())

        for level in levels:
            summary += '{}-EM Models: {} \n'.format(level, self.total_per_level(level))

            for (model, yn) in zip(self.class_models[level], self.class_models_yn[level]):
                if yn:
                    if level == 2:
                        model_string = self.unique_classes[model]
                    else:
                        model_string = '-'.join(self.unique_classes[np.array(model)])
                    summary += '  {}: {} \n'.format(model_string, self.total_per_class_model(model))
            summary += '\n'

        return summary

    def save(self) -> str:
        """
        Get a summary of the selected models, designed specifically for the 'SAVE' functionality in the MESMA gui.

        :return: A summary of the selected models, specifically to save in the MESMA settings [multi-line-str].
        """

        summary = 'x-EM Levels:\n'
        summary += ', '.join(map(str, self.level_yn))
        summary += '\n'

        summary += 'classes per x-EM Level:\n'
        for level in self.class_per_level.keys():
            summary += '{}-EMs: {}\n'.format(level, ', '.join(map(str, self.class_per_level[level])))

        summary += 'models per x-EM Level:\n'
        for level in self.class_models_yn.keys():
            summary += '{}-EMs: {}\n'.format(level, ', '.join(map(str, self.class_models_yn[level])))

        return summary