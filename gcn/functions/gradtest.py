import torch
from torch.autograd import Variable
from gcn.functions import GOF_Function, my_GOF_Function
from torch.autograd import gradcheck

def cpu_gpu_check():
    gof = GOF_Function.apply
    # gof = my_GOF_Function()

    weight = torch.rand(2,2,4,3,3).double()
    gfb = torch.rand(4,3,3).double()

    weight_cpu = Variable(weight, requires_grad=True)
    weight_gpu = Variable(weight.cuda(), requires_grad=True)

    gfb_cpu = Variable(gfb, requires_grad=False)
    gfb_gpu = Variable(gfb.cuda(), requires_grad=False)

    # Forward results checking...
    print('-'*80)
    output_cpu = gof(weight_cpu, gfb_cpu)
    output_gpu = gof(weight_gpu, gfb_gpu)
    if torch.equal(output_cpu, output_gpu.cpu()):
        print("Forward results do agree!")
    else:
        print("Forward results do not agree!")
        print('Results on cpu:', output_cpu)
        print('Results on gpu:', output_gpu)
    
    # Backward results checking...
    print('-'*80)
    output_cpu.backward(torch.ones(output_cpu.size()).double())
    output_gpu.backward(torch.ones(output_gpu.size()).double().cuda())
    if torch.equal(weight_cpu.grad, weight_gpu.grad.cpu()):
        print("Backward grads do agree!")
    else:
        print("Backward grads do not agree!")
        print('Grad on cpu:', weight_cpu.grad)
        print('Grad on gpu:', weight_gpu.grad)

    # Gradcheck on cpu
    print('-'*80)
    print('Gradcheck on cpu:')
    inputs_cpu = (weight_cpu, gfb_cpu)
    test_cpu = gradcheck(gof, inputs_cpu, eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=True)
    print(test_cpu)

    # Gradcheck on gpu
    print('-'*80)
    print('Gradcheck on gpu:')
    inputs_gpu = (weight_gpu, gfb_gpu)
    test_gpu = gradcheck(gof, inputs_gpu, eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=True)
    print(test_gpu)

if __name__ == "__main__":
    cpu_gpu_check()