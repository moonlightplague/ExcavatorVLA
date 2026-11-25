import unittest
from types import SimpleNamespace
import torch
from lerobot.policies.act.command_pose_extension import CommandPoseMemory


class PoseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.config=SimpleNamespace(chunk_size=10,robot_state_feature=SimpleNamespace(shape=(28,)),
            command_pose_center=[0.]*4,command_pose_radius=[2.,1.,2.,3.],
            state_normalization_mean=[0.]*28,state_normalization_std=[1.]*28,
            action_normalization_mean=[.1]*4,action_normalization_std=[.5]*4)
        self.memory=CommandPoseMemory(1604,16,40,self.config).eval()
        self.feature=torch.randn(2,9,1604)*.1

    def test_position_action_roundtrip(self):
        action=torch.randn(2,9,10,4)*.1
        poses=self.memory.target_poses(self.feature,action)
        torch.testing.assert_close(self.memory.decode(self.feature,poses),action,rtol=1e-5,atol=2e-6)

    def test_sequence_matches_online(self):
        expected=self.memory(self.feature)[0];hidden=None;actual=[]
        for item in self.feature.unbind(1):
            value,hidden=self.memory(item[:,None],hidden);actual.append(value)
        torch.testing.assert_close(torch.cat(actual,1),expected,atol=2e-5,rtol=1e-5)

    def test_pose_range_and_parameterized_action(self):
        extreme=self.feature.clone();extreme[...,:4]=-3.14159;extreme[...,8:12]=0
        poses,_=self.memory.predict_poses(extreme)
        self.assertTrue((poses.abs()<=self.memory.pose_radius+1e-6).all())
        delta,_=self.memory(extreme)
        normalized=delta+extreme[...,-40:]
        velocity=normalized.reshape(2,9,10,4)*self.memory.action_std+self.memory.action_mean
        self.assertTrue((velocity[...,0,0]>0).all())
        torch.testing.assert_close(velocity,self.memory.decode(extreme,poses)*self.memory.action_std+self.memory.action_mean,atol=1e-5,rtol=1e-5)

    def test_finite_pose_and_velocity_learning(self):
        self.memory.train();optimizer=torch.optim.Adam(self.memory.parameters(),lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad();delta,_=self.memory(self.feature)
            loss=(delta+self.feature[...,-40:]).abs().mean();loss.backward()
            self.assertTrue(torch.isfinite(loss));optimizer.step()
        self.assertGreater(self.memory.pose_state.weight.grad.abs().sum().item(),0)
        self.assertGreater(self.memory.gru.weight_ih_l0.grad.abs().sum().item(),0)


if __name__=='__main__':unittest.main()
