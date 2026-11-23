"""Realized native convolution consumers do not create stranded FP8 edges."""

import unittest
from copy import deepcopy

import torch
from torch import nn

from deepopt import FP8Config
from deepopt.opt.policy import prepare_fp8_model
from deepopt.opt.precision import precision_report


class RuntimeSpeedPlanningTests(unittest.TestCase):
    def test_native_spatial_removes_producer_island_through_nonlinearity(self):
        for activation in (nn.ReLU, nn.SiLU):
            for policy in ('strict', 'native_spatial'):
                model = nn.Sequential(nn.Conv2d(4,4,3,padding=1), nn.BatchNorm2d(4), activation(),
                                      nn.Conv2d(4,4,3,padding=1))
                prepared, _ = prepare_fp8_model(model, FP8Config(enabled=True, backend='reference',
                                               trace_mode='modules', convolution_policy=policy))
                report = precision_report(prepared)
                self.assertFalse(report['fallbacks'])
                fp8 = [edge for edge in report['edges'] if edge['precision']=='fp8']
                if policy == 'strict':
                    self.assertTrue(fp8)
                else:
                    self.assertFalse(fp8)
                    self.assertEqual(prepared[1]._precision_outputs[True], (None,))

    def test_native_spatial_pointwise_consumer_keeps_fp8_edge(self):
        for groups, stride, expected in ((1,1,True), (4,1,False), (1,2,False)):
            model=nn.Sequential(nn.BatchNorm2d(4),nn.ReLU(),nn.Conv2d(4,4,1,stride=stride,groups=groups))
            prepared,_=prepare_fp8_model(model,FP8Config(enabled=True,backend='reference',trace_mode='modules',
                                                        convolution_policy='native_spatial'))
            report=precision_report(prepared)
            self.assertFalse(report['fallbacks'])
            self.assertEqual(any(e['precision']=='fp8' for e in report['edges']), expected)

    def test_functional_weight_metadata_resolves_native_consumer(self):
        class Functional(nn.Module):
            def __init__(self,kernel):
                super().__init__();self.bn=nn.BatchNorm2d(4);self.weight=nn.Parameter(torch.randn(4,4,kernel,kernel))
            def forward(self,x):
                return torch.nn.functional.conv2d(torch.relu(self.bn(x)),self.weight)
        for kernel in (1,3):
            prepared,_=prepare_fp8_model(Functional(kernel),FP8Config(enabled=True,backend='reference',trace_mode='full',
                intercept_linear=True,convolution_policy='native_spatial'))
            report=precision_report(prepared)
            self.assertFalse(report['fallbacks'])
            self.assertEqual(any(e['precision']=='fp8' for e in report['edges']),kernel==1)

    def test_native_island_preserves_outputs_and_parameter_gradients(self):
        torch.manual_seed(14)
        source=nn.Sequential(nn.BatchNorm2d(4),nn.SiLU(),nn.Conv2d(4,4,3,padding=1))
        control=deepcopy(source)
        prepared,_=prepare_fp8_model(source,FP8Config(enabled=True,backend='reference',trace_mode='modules',
                                                     convolution_policy='native_spatial'))
        x=torch.randn(2,4,5,7,requires_grad=True); other=x.detach().clone().requires_grad_()
        actual=prepared(x); expected=control(other)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        actual.square().sum().backward();expected.square().sum().backward()
        torch.testing.assert_close(x.grad,other.grad,rtol=2e-5,atol=1e-5)
        for a,b in zip(prepared.parameters(),control.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=2e-5,atol=1e-5)


if __name__=='__main__':
    unittest.main()
