> 拖了好久的Transformer，终于理解别人的代码，也尝试自己手写一下。有些地方还是得自己亲手写了才知道啊。
> 难点感觉在data flow的维度上，注意把batch_size，n_heads暂时忽视，只看一个句子来分析数据流会更好理解吧。
> 17年的模型，火了这么久，现在才基本弄懂是什么个原理，唉，还是太菜了啊。自己还是不太适合走算法这条路，写代码的耐心不够，写着写着就分心，不耐烦。所以，最后工作的方向是什么呢？还是得继续踩坑，不断寻找啊！


## transformer.ipynb
参考一个大佬博主的博客，加上他b站录制的视频，训练自己手写transformer的代码，适合学习，各数据维度都有写上，清晰明了。强烈推荐！！！

[博客连接](https://wmathor.com/index.php/archives/1455/)

[b站连接](https://space.bilibili.com/181990557/channel/seriesdetail?sid=216163)

## transformer-New-Year-main

利用transformer做的一个基础小项目，学习所得：

- 如何写tokenizer
- 如何使用transformer，关键点：输入和输出、训练与预测时的区别、预测的时候怎么获取最后的输出。



之前看的博主的transformer和这个项目的预测部分做了下对比，发现预测输出其实本质是一样的，就是方式有点不同。这个项目里是：每次更新dec_inputs，然后得到dec_outputs里取最后一维度的预测值作为新的dec_input，拼接到dec_inputs里。最后把整个dec_inputs作为整体重新输入模型做预测，就得到真正的预测值了。