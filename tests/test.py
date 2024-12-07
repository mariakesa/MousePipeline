import unittest
import numpy as np

# Example minimal STAProcessEID with compute_sta function
class STAProcessEID:
    def __init__(self, embedding_dim):
        # embedding_dim is the number of dimensions in the embedding vectors
        self.embedding_dim = embedding_dim

    def compute_sta(self, events, n_neurons, embedding_trial):
        """
        Compute STA increments from events for a given trial.

        Parameters
        ----------
        events : tuple of (neuron_indices, time_indices)
            Indices of neurons and their event times within the trial.
        n_neurons : int
            Total number of neurons.
        embedding_trial : np.ndarray
            Embedding vectors for the trial's time points. Shape: (trial_time_points, embedding_dim)

        Returns
        -------
        sta_increment : np.ndarray
            Sum of embeddings at event times for each neuron. Shape: (n_neurons, embedding_dim)
        count_increment : np.ndarray
            Number of events per neuron. Shape: (n_neurons,)
        """
        neuron_indices, time_indices = events

        sta_increment = np.zeros((n_neurons, self.embedding_dim))
        count_increment = np.zeros(n_neurons, dtype=int)

        if len(neuron_indices) > 0:
            sta_increment[neuron_indices] += embedding_trial[time_indices]
            counts = np.bincount(neuron_indices, minlength=n_neurons)
            count_increment += counts

        return sta_increment, count_increment


class TestComputeSTA(unittest.TestCase):
    def setUp(self):
        # For testing purposes, assume embedding_dim=3
        self.embedding_dim = 3
        self.sta_processor = STAProcessEID(self.embedding_dim)

    def test_no_events(self):
        """Test compute_sta when there are no spike events."""
        n_neurons = 3
        events = (np.array([]), np.array([]))  # No events
        # embedding_trial: Arbitrary shape/timepoints, let's say (3, 3)
        embedding_trial = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])

        sta_increment, count_increment = self.sta_processor.compute_sta(events, n_neurons, embedding_trial)

        expected_sta_increment = np.zeros((n_neurons, self.embedding_dim))
        expected_count_increment = np.zeros(n_neurons, dtype=int)

        np.testing.assert_array_equal(sta_increment, expected_sta_increment, 
                                      "With no events, STA increment should remain all zeros.")
        np.testing.assert_array_equal(count_increment, expected_count_increment, 
                                      "With no events, count increment should remain all zeros.")

    def test_single_event(self):
        """Test compute_sta with a single event for one neuron."""
        n_neurons = 3
        # Suppose neuron 1 spiked at time index 1
        events = (np.array([1]), np.array([1]))
        embedding_trial = np.array([
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0]
        ])

        sta_increment, count_increment = self.sta_processor.compute_sta(events, n_neurons, embedding_trial)

        expected_sta_increment = np.zeros((n_neurons, self.embedding_dim))
        expected_sta_increment[1] = [20.0, 21.0, 22.0]  # Neuron 1 had one spike at time 1
        expected_count_increment = np.array([0, 1, 0])

        np.testing.assert_array_equal(sta_increment, expected_sta_increment,
                                      "Single event should add the correct embedding vector to the neuron that spiked.")
        np.testing.assert_array_equal(count_increment, expected_count_increment,
                                      "Count increment should reflect one spike for neuron 1.")

    def test_multiple_events(self):
        """Test compute_sta with multiple events across neurons."""
        n_neurons = 3
        # Suppose neuron 0 spikes at times 0,2 and neuron 2 spikes at time 1
        events = (np.array([0, 0, 2]), np.array([0, 2, 1]))
        embedding_trial = np.array([
            [1.0,  2.0,  3.0],  # time 0
            [4.0,  5.0,  6.0],  # time 1
            [7.0,  8.0,  9.0]   # time 2
        ]).T

        sta_increment, count_increment = self.sta_processor.compute_sta(events, n_neurons, embedding_trial)

        # Neuron 0: events at time 0 and time 2
        expected_sta_increment = np.zeros((n_neurons, self.embedding_dim))
        expected_sta_increment[0] = [1.0+7.0, 2.0+8.0, 3.0+9.0]  # (8.0,10.0,12.0)
        # Neuron 2: event at time 1
        expected_sta_increment[2] = [4.0, 5.0, 6.0]

        expected_count_increment = np.array([2, 0, 1])  # Neuron 0 had 2 events, Neuron 2 had 1 event

        np.testing.assert_array_equal(sta_increment, expected_sta_increment,
                                      "STA increments should correctly accumulate embeddings for multiple events.")
        np.testing.assert_array_equal(count_increment, expected_count_increment,
                                      "Count increments should correctly tally the number of events per neuron.")

if __name__ == '__main__':
    unittest.main()
