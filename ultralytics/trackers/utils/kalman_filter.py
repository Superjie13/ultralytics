# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import scipy.linalg


class KalmanFilterXYAH:
    """
    A KalmanFilterXYAH class for tracking bounding boxes in image space using a Kalman filter.

    Implements a simple Kalman filter for tracking bounding boxes in image space. The 8-dimensional state space
    (x, y, a, h, vx, vy, va, vh) contains the bounding box center position (x, y), aspect ratio a, height h, and their
    respective velocities. Object motion follows a constant velocity model, and bounding box location (x, y, a, h) is
    taken as a direct observation of the state space (linear observation model).

    Attributes:
        _motion_mat (np.ndarray): The motion matrix for the Kalman filter.
        _update_mat (np.ndarray): The update matrix for the Kalman filter.
        _std_weight_position (float): Standard deviation weight for position.
        _std_weight_velocity (float): Standard deviation weight for velocity.

    Methods:
        initiate: Creates a track from an unassociated measurement.
        predict: Runs the Kalman filter prediction step.
        project: Projects the state distribution to measurement space.
        multi_predict: Runs the Kalman filter prediction step (vectorized version).
        update: Runs the Kalman filter correction step.
        gating_distance: Computes the gating distance between state distribution and measurements.

    Examples:
        Initialize the Kalman filter and create a track from a measurement
        >>> kf = KalmanFilterXYAH()
        >>> measurement = np.array([100, 200, 1.5, 50])
        >>> mean, covariance = kf.initiate(measurement)
        >>> print(mean)
        >>> print(covariance)
    """

    def __init__(self):
        """
        Initialize Kalman filter model matrices with motion and observation uncertainty weights.

        The Kalman filter is initialized with an 8-dimensional state space (x, y, a, h, vx, vy, va, vh), where (x, y)
        represents the bounding box center position, 'a' is the aspect ratio, 'h' is the height, and their respective
        velocities are (vx, vy, va, vh). The filter uses a constant velocity model for object motion and a linear
        observation model for bounding box location.

        Examples:
            Initialize a Kalman filter for tracking:
            >>> kf = KalmanFilterXYAH()
        """
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current state estimate
        self._std_weight_position = 8.0 / 20
        self._std_weight_velocity = 8.0 / 160

    def initiate(self, measurement: np.ndarray):
        """
        Create a track from an unassociated measurement.

        Args:
            measurement (np.ndarray): Bounding box coordinates (x, y, a, h) with center position (x, y), aspect ratio a,
                and height h.

        Returns:
            (np.ndarray): Mean vector (8-dimensional) of the new track. Unobserved velocities are initialized to 0 mean.
            (np.ndarray): Covariance matrix (8x8 dimensional) of the new track.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> measurement = np.array([100, 50, 1.5, 200])
            >>> mean, covariance = kf.initiate(measurement)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """
        Run Kalman filter prediction step.

        Args:
            mean (np.ndarray): The 8-dimensional mean vector of the object state at the previous time step.
            covariance (np.ndarray): The 8x8-dimensional covariance matrix of the object state at the previous time step.

        Returns:
            (np.ndarray): Mean vector of the predicted state. Unobserved velocities are initialized to 0 mean.
            (np.ndarray): Covariance matrix of the predicted state.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """
        Project state distribution to measurement space.

        Args:
            mean (np.ndarray): The state's mean vector (8 dimensional array).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (np.ndarray): Projected mean of the given state estimate.
            (np.ndarray): Projected covariance matrix of the given state estimate.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> projected_mean, projected_covariance = kf.project(mean, covariance)
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """
        Run Kalman filter prediction step for multiple object states (Vectorized version).

        Args:
            mean (np.ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (np.ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            (np.ndarray): Mean matrix of the predicted states with shape (N, 8).
            (np.ndarray): Covariance matrix of the predicted states with shape (N, 8, 8).

        Examples:
            >>> mean = np.random.rand(10, 8)  # 10 object states
            >>> covariance = np.random.rand(10, 8, 8)  # Covariance matrices for 10 object states
            >>> predicted_mean, predicted_covariance = kalman_filter.multi_predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """
        Run Kalman filter correction step.

        Args:
            mean (np.ndarray): The predicted state's mean vector (8 dimensional).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (np.ndarray): The 4 dimensional measurement vector (x, y, a, h), where (x, y) is the center
                position, a the aspect ratio, and h the height of the bounding box.

        Returns:
            (np.ndarray): Measurement-corrected state mean.
            (np.ndarray): Measurement-corrected state covariance.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurement = np.array([1, 1, 1, 1])
            >>> new_mean, new_covariance = kf.update(mean, covariance, measurement)
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If `only_position` is False, the chi-square
        distribution has 4 degrees of freedom, otherwise 2.

        Args:
            mean (np.ndarray): Mean vector over the state distribution (8 dimensional).
            covariance (np.ndarray): Covariance of the state distribution (8x8 dimensional).
            measurements (np.ndarray): An (N, 4) matrix of N measurements, each in format (x, y, a, h) where (x, y) is the
                bounding box center position, a the aspect ratio, and h the height.
            only_position (bool): If True, distance computation is done with respect to box center position only.
            metric (str): The metric to use for calculating the distance. Options are 'gaussian' for the squared
                Euclidean distance and 'maha' for the squared Mahalanobis distance.

        Returns:
            (np.ndarray): Returns an array of length N, where the i-th element contains the squared distance between
                (mean, covariance) and `measurements[i]`.

        Examples:
            Compute gating distance using Mahalanobis metric:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurements = np.array([[1, 1, 1, 1], [2, 2, 1, 1]])
            >>> distances = kf.gating_distance(mean, covariance, measurements, only_position=False, metric="maha")
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)  # square maha
        else:
            raise ValueError("Invalid distance metric")


class KalmanFilterXYWH(KalmanFilterXYAH):
    """
    A KalmanFilterXYWH class for tracking bounding boxes in image space using a Kalman filter.

    Implements a Kalman filter for tracking bounding boxes with state space (x, y, w, h, vx, vy, vw, vh), where
    (x, y) is the center position, w is the width, h is the height, and vx, vy, vw, vh are their respective velocities.
    The object motion follows a constant velocity model, and the bounding box location (x, y, w, h) is taken as a direct
    observation of the state space (linear observation model).

    Attributes:
        _motion_mat (np.ndarray): The motion matrix for the Kalman filter.
        _update_mat (np.ndarray): The update matrix for the Kalman filter.
        _std_weight_position (float): Standard deviation weight for position.
        _std_weight_velocity (float): Standard deviation weight for velocity.

    Methods:
        initiate: Creates a track from an unassociated measurement.
        predict: Runs the Kalman filter prediction step.
        project: Projects the state distribution to measurement space.
        multi_predict: Runs the Kalman filter prediction step in a vectorized manner.
        update: Runs the Kalman filter correction step.

    Examples:
        Create a Kalman filter and initialize a track
        >>> kf = KalmanFilterXYWH()
        >>> measurement = np.array([100, 50, 20, 40])
        >>> mean, covariance = kf.initiate(measurement)
        >>> print(mean)
        >>> print(covariance)
    """

    def initiate(self, measurement: np.ndarray):
        """
        Create track from unassociated measurement.

        Args:
            measurement (np.ndarray): Bounding box coordinates (x, y, w, h) with center position (x, y), width, and height.

        Returns:
            (np.ndarray): Mean vector (8 dimensional) of the new track. Unobserved velocities are initialized to 0 mean.
            (np.ndarray): Covariance matrix (8x8 dimensional) of the new track.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> measurement = np.array([100, 50, 20, 40])
            >>> mean, covariance = kf.initiate(measurement)
            >>> print(mean)
            [100.  50.  20.  40.   0.   0.   0.   0.]
            >>> print(covariance)
            [[ 4.  0.  0.  0.  0.  0.  0.  0.]
             [ 0.  4.  0.  0.  0.  0.  0.  0.]
             [ 0.  0.  4.  0.  0.  0.  0.  0.]
             [ 0.  0.  0.  4.  0.  0.  0.  0.]
             [ 0.  0.  0.  0.  0.25  0.  0.  0.]
             [ 0.  0.  0.  0.  0.  0.25  0.  0.]
             [ 0.  0.  0.  0.  0.  0.  0.25  0.]
             [ 0.  0.  0.  0.  0.  0.  0.  0.25]]
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Run Kalman filter prediction step.

        Args:
            mean (np.ndarray): The 8-dimensional mean vector of the object state at the previous time step.
            covariance (np.ndarray): The 8x8-dimensional covariance matrix of the object state at the previous time step.

        Returns:
            (np.ndarray): Mean vector of the predicted state. Unobserved velocities are initialized to 0 mean.
            (np.ndarray): Covariance matrix of the predicted state.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """
        Project state distribution to measurement space.

        Args:
            mean (np.ndarray): The state's mean vector (8 dimensional array).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (np.ndarray): Projected mean of the given state estimate.
            (np.ndarray): Projected covariance matrix of the given state estimate.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> projected_mean, projected_cov = kf.project(mean, covariance)
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """
        Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (np.ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (np.ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            (np.ndarray): Mean matrix of the predicted states with shape (N, 8).
            (np.ndarray): Covariance matrix of the predicted states with shape (N, 8, 8).

        Examples:
            >>> mean = np.random.rand(5, 8)  # 5 objects with 8-dimensional state vectors
            >>> covariance = np.random.rand(5, 8, 8)  # 5 objects with 8x8 covariance matrices
            >>> kf = KalmanFilterXYWH()
            >>> predicted_mean, predicted_covariance = kf.multi_predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        Run Kalman filter correction step.

        Args:
            mean (np.ndarray): The predicted state's mean vector (8 dimensional).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (np.ndarray): The 4 dimensional measurement vector (x, y, w, h), where (x, y) is the center
                position, w the width, and h the height of the bounding box.

        Returns:
            (np.ndarray): Measurement-corrected state mean.
            (np.ndarray): Measurement-corrected state covariance.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurement = np.array([0.5, 0.5, 1.2, 1.2])
            >>> new_mean, new_covariance = kf.update(mean, covariance, measurement)
        """
        return super().update(mean, covariance, measurement)


class KalmanFilterXY:
    def __init__(self, dt=1):
        ndim = 4
        self.gating_threshold = 10

        # state transition matrix
        # state: [x, y, vx, vy]
        self._motion_mat = np.eye(ndim, dtype=np.float32)
        self._motion_mat[0, 2] = dt
        self._motion_mat[1, 3] = dt
        # measurement matrix
        # measurement: [x, y]
        self._update_mat = np.eye(ndim, dtype=np.float32)[:2]

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 1
        self._std_weight_velocity = 1. / 4

    def initiate(self, measurement: np.array):
        """Crate track from unassociated measurement
        Params:
            measurement (ndarray): cluster center (x, y). 
        Return:
            (ndarray, ndarray): Returns the mean vector (4 dimensions) and
            covariance matrix (4x4 dimensions) of the new track.
            Note: unobserved velocities are initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Initialize the state covariance
        std_x = self._std_weight_position
        std_y = self._std_weight_position
        std_vx = self._std_weight_velocity
        std_vy = self._std_weight_velocity

        covariance = np.diag(np.square([std_x, std_y, std_vx, std_vy]))
        return mean, covariance
    
    def predict(self, mean: np.array, covariance: np.array):
        """Run Kalman filter prediction step.
        Params:
            mean (np.ndarray): The 4 dimensional mean vector of the object state at the previous time step.
            covariance (np.ndarray): The 4x4 dimensional covariance matrix of the object state at the previous time step.
        Return:
            (ndarray, ndarray): Returns the mean vector and covariance matrix of the predicted state.
        """
        std_x = self._std_weight_position
        std_y = self._std_weight_position
        std_vx = self._std_weight_velocity
        std_vy = self._std_weight_velocity

        # Process noise
        motion_cov = np.diag(np.square([std_x, std_y, std_vx, std_vy]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov  # P_k = F * P_k-1 * F^T + Q
        return mean, covariance
    
    def project(self, mean: np.array, covariance: np.array):
        """Project the state distribution to measurement space.
        Params:
            mean (ndarray): The state's mean vector (4 dimensions).
            covariance (ndarray): The state's covariance matrix (4x4 dimensions).
        Return:
            (ndarray, ndarray): Returns the projected mean and covariance matrix of the given state estimate.
        """
        std_x = 0.5 * self._std_weight_position
        std_y = 0.5 * self._std_weight_position
        # measurement noise
        innovation_cov = np.diag(np.square([std_x, std_y]))

        mean = np.dot(self._update_mat, mean)  # Z_k = H * X_k
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))  # S_k = H * P_k * H^T
        return mean, covariance + innovation_cov
    
    def update(self, mean: np.array, covariance: np.array, measurement: np.array):
        """Run Kalman filter correction step.
        Params:
            mean (ndarray): The predicted state's mean vector (4 dimensions).
            covariance (ndarray): The predicted state's covariance matrix (4x4 dimensions).
            measurement (ndarray): The 2 dimensional measurement (x, y).
        Return:
            (ndarray, ndarray): Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(kalman_gain, innovation)
        new_cov = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_cov
    
    def gating_distance(self, mean: np.array, covariance: np.array, measurements: np.array):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`.
        Params:
            mean (ndarray): The state's mean vector (4 dimensions).
            covariance (ndarray): The state's covariance matrix (4x4 dimensions).
            measurements (ndarray): The 2 dimensional measurement (x, y). 
                shape: (n, 2). where n is the number of clusters.
        Return:
            (ndarray): Returns an array of length m, where the i-th element contains the squared Mahalanobis distance
            between (mean, covariance) and `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        cholesk_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(cholesk_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
    
    def compute_distance(self, mean: np.array, measurements: np.array):
        """Compute the distance between the track mean and the measurements.
        Params:
            mean (ndarray): The track's mean vector (4 dimensions).
            measurements (ndarray): The 2 dimensional measurements (x, y).
                shape: (n, 2). where n is the number of clusters.
        Return:
            (ndarray): Return an array of length n, where the i-th element contains the Euclidean distance
            between the track's mean and `measurements[i]`.
        """
        mean = mean[:2]
        d = measurements - mean
        distance = np.sqrt(np.sum(d * d, axis=1))
        return distance
    
    def track_object(self, track_mean, track_covariance, measurements: np.array, thr=2):
        _mean, _covariance = self.predict(track_mean, track_covariance)
        # cost = self.gating_distance(_mean, _covariance, measurements)
        # cost[cost > self.gating_threshold] = np.nan
        cost = self.compute_distance(_mean, measurements)
        cost[cost > thr] = np.nan
        return _mean, _covariance, cost

    def futures(self, mean: np.array, covariance: np.array, n=10):
        """Predict the future states
        Params:
            mean (ndarray): The state's mean vector (4 dimensions).
            covariance (ndarray): The state's covariance matrix (4x4 dimensions).
            n (int): number of future states
        Return:
            (ndarray): Returns the future states
        """
        states = []
        for i in range(n):
            mean, covariance = self.predict(mean, covariance)
            states.append(mean)
        return np.array(states)