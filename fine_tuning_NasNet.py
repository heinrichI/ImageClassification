model.trainable = True

set_trainable = False
for layer in model.layers:
  if layer.name == 'activation_253':
    set_trainable = True
  if set_trainable:
    layer.trainable = True
  else:
    layer.trainable = False
  print("layer {} is {}".format(layer.name, '+++trainable' if layer.trainable else '---frozen'))