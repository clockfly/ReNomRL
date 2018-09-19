Introduction
============

**blahblah**
..
.. image:: /_static/image/top.png

1. blahblah
----------

.. blahblah.

The concept of ReNomRL is **blahblah**.
..
Recent developing deep learning technology realizes extremely big improvement at
recognition accuracy.  
..
However if you would create a recognition model for any business scene such as 
recognising damages of manufactured products, there are still many problems for 
earning high accuracy recognition model.
..
For example, correcting training dataset, programming recognition model and train it, 
evaluating the model, and so on.
..
Especially, even if deep learning era, it is required to tune up the hyper parameters of 
the recognition model. It requires many try and errors.
..
ReNomIMG allows you to build object detection model easily.

2. blahblah
-------------------------------

ReNomIMG provides gui tool and python api.

GUI tool
~~~~~~~~~~~~~~

ReNomIMG GUI tool allows you to build object detection models.
What users have to do are **preparing training data**, 
**setting train configuration** and **pushing run button**.


.. 下の図は, 後で差し替え

.. 
image:: /_static/image/renomimg_gui_top.png


Python API
~~~~~~~~~~~
ReNomIMG API is a python api which provides you not only modern **object detection model** 
but also **classification model**, **segmentation model**. 

And more, all those models have pretrained weights.
This makes models more accurate one.

An example code is bellow. Using ReNomIMG, you can build a model and train it in 3 lines.
..
**Building a VGG16 Model**
..
.. code-block :: python
    :linenos:
    :emphasize-lines: 12,13,16
..
    from renom_img.api.classification.vgg import VGG16
    from renom_img.api.utility.load import parse_xml_detection
    from renom_img.api.utility.misc.display import draw_box
..
    ## Data preparation.
    train_image_path_list = ...
    train_label_list = ...
    valid_image_path_list = ...
    valid_label_list = ...
..
    ## Build a classification model(ex: VGG16).
    model = VGG16(class_map, load_pretrained_weight=True, train_whole_network=False)
    model.fit(train_image_path_list, train_label_list, valid_image_path_list, valid_label_list)
..
    ## Prediction.
    prediction = model.predict(new_image)

