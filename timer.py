
class Timer:

    def __init__(self, tag_buf):
        self.tag_buf = tag_buf
        self.time = dict()
        self.reset()


    def set(self, time, tag):
        self.time[tag] = time

    def run(self, time, tag):
        self.time[tag] += time

    def reset(self):
        for tag in self.tag_buf:
            self.time[tag] = 0


global_timer = Timer(['dct', 'idct'])

    # t3 = time.time()
# output = model(torch_img)
# torch.cuda.synchronize()
# t4 = time.time()
# print(t4 - t3)