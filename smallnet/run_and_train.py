import caffe
net = caffe.Net('.prototxt', caffe.TEST)
solver = caffe.get_solver('solver_smallnet.prototxt')

solver.net.forward()
solver.net.backward()
solver.step(1)
solver.solve()
