
def heap_create(new_list):

    if len(new_list) > 0:
        for i in range((len(new_list) - 1) // 2, -1, -1):
            adjust_sub_heap(new_list, i)

    return new_list


def heap_push(heap_list, node):
    """ 插入新的节点值 """
    # 在数组的末尾插入新的节点值
    heap_list.append(node)

    # 调整该节点所在的父节点及其父节点的堆
    father_node_index = len(heap_list) - 1
    while father_node_index >= 0:
        father_node_index = (father_node_index - 1) // 2
        # 插入时，当该节点所在的最大堆不需要调整时，则不需要进一步调整其父节点的最大堆
        if not adjust_sub_heap(heap_list, father_node_index):
            break


def heap_pop(heap_list) -> int:
    """ 删除并返回最大的节点值 """
    # 删除顶点节点并保存最大的节点值
    root_node = heap_list[0]

    # 只有一个节点值时，清空最大堆并返回最大值
    if len(heap_list) == 1:
        heap_list.clear()
        return root_node

    # 将末尾的节点移至头结点,并删除尾结点
    heap_list[0] = heap_list[len(heap_list) - 1]
    del heap_list[len(heap_list) - 1]

    # 调整最大堆
    adjust_sub_heap(heap_list, 0)

    # 返回最大值
    return root_node


def adjust_sub_heap(heap_list, top_node_index: int) -> bool:
    """
    调整以某节点的顶点的最大堆
    :param top_node_index: 节点所在的序号
    :return: 是否调整过以该节点为顶点的最大堆
    """
    # 当编号小于0或者无子节点时，不需要调整以该节点为顶点的最大堆
    if top_node_index < 0 or top_node_index > (len(heap_list) - 1) // 2:
        return False

    left_node_index = 2 * top_node_index + 1
    right_node_index = 2 * top_node_index + 2

    # 获取顶节点、左节点和右节点中的最大数字节点的位置
    # 因为只要求顶节点大于孩子节点，不要求左孩子节点大于右孩子节点，所以只需要和左右孩子节点中较大的那个交换即可
    max_num_index = top_node_index
    if left_node_index < len(heap_list) and \
            heap_list[max_num_index][0] < heap_list[left_node_index][0]:
        max_num_index = left_node_index
    if right_node_index < len(heap_list) and \
            heap_list[max_num_index][0] < heap_list[right_node_index][0]:
        max_num_index = right_node_index

    # print("%s: %d, %d" % (str(heap_list), top_node_index, max_num_index))

    # 调整堆和子堆的位置，因为小的数字往下移，可能导致子堆不满足最大堆的规定
    if max_num_index != top_node_index:
        swap_heap_node(heap_list, max_num_index, top_node_index)
        adjust_sub_heap(heap_list, max_num_index)
        return True

    return False


def swap_heap_node(heap_list, a_node_index: int, b_node_index: int):
    """ 交换两个节点 """
    temp = heap_list[a_node_index]
    heap_list[a_node_index] = heap_list[b_node_index]
    heap_list[b_node_index] = temp





# x_list = [(5, "5", "5"), (7, "7", "7"), (3, "3", "3")]
# heap_list = heap_create(x_list)
# print(heap_pop(heap_list))
# print(heap_pop(heap_list))
# print(heap_pop(heap_list))
#
# heap_push(heap_list, (2.1, "2", "2"))
# heap_push(heap_list, (20, "20", "20"))
# heap_push(heap_list, (-20.5, "-20", "-20"))
#
# print(heap_pop(heap_list))
# print(heap_pop(heap_list))
# print(heap_pop(heap_list))

