# Importing required libraries for testing
import unittest
import torch
import numpy as np
from ai import Network, Dqn, ReplayMemory
from map import CarApp, MyPaintWidget


# Front End (UI) Tests
class TestFrontEnd(unittest.TestCase):

    def setUp(self):
        self.app = CarApp()
        self.app.build()

    def test_canvas_drawing(self):
        # Test if the canvas can be cleared properly
        self.app.clear_canvas(None)
        self.assertTrue(np.all(self.app.painter.sand == 0))

    def test_button_functionality(self):
        # Test save and load buttons
        self.app.save(None)  # Check if it runs without errors
        self.app.load(None)  # Check if it runs without errors


# Machine Learning (Deep Q-Learning) Tests
class TestDQN(unittest.TestCase):

    def setUp(self):
        self.input_size = 5  # Example input size
        self.nb_action = 3  # Example number of actions
        self.dqn = Dqn(input_size=self.input_size, nb_action=self.nb_action, gamma=0.9)

    def test_neural_network_forward_pass(self):
        # Creating a dummy state tensor
        state = torch.Tensor([1.0, 0.0, 0.0, 1.0, 0.0]).unsqueeze(0)
        # Forward pass through the network
        q_values = self.dqn.model.forward(state)
        self.assertEqual(q_values.size(), (1, self.nb_action))

    def test_action_selection(self):
        # Creating a dummy state tensor
        state = torch.Tensor([1.0, 0.0, 0.0, 1.0, 0.0]).unsqueeze(0)
        # Select an action
        action = self.dqn.select_action(state)
        self.assertIn(action, range(self.nb_action))

    def test_learning_process(self):
        # Dummy data for learning
        batch_state = torch.randn((10, self.input_size))
        batch_next_state = torch.randn((10, self.input_size))
        batch_action = torch.randint(0, self.nb_action, (10, 1))
        batch_reward = torch.randn((10, 1))
        # Run the learn function
        self.dqn.learn(batch_state, batch_next_state, batch_reward, batch_action)
        # There are no explicit output assertions; this checks if the function runs without error


# Database (Experience Replay Memory) Tests
class TestReplayMemory(unittest.TestCase):

    def setUp(self):
        self.memory = ReplayMemory(capacity=100)

    def test_memory_capacity(self):
        # Add dummy transitions
        for i in range(150):
            self.memory.push((torch.randn(5), torch.tensor([i]), torch.randn(1)))
        # Check if the capacity is not exceeded
        self.assertEqual(len(self.memory.memory), 100)

    def test_sample_function(self):
        # Add dummy transitions
        for i in range(50):
            self.memory.push((torch.randn(5), torch.tensor([i]), torch.randn(1)))
        # Sample a batch
        batch = list(self.memory.sample(10))
        self.assertEqual(len(batch), 3)  # Should return three elements: states, actions, rewards
        for element in batch:
            self.assertEqual(len(element), 10)  # Each element should have 10 entries


# Running all tests
if __name__ == '__main__':
    unittest.main()
