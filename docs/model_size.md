# Model size accounting

The default Shorkie LM has two related but distinct size counts:

- **13,651,812 trainable parameters**: the sum over `model.parameters()`.
- **13,665,856 state elements in 356 tensors**: the sum over the complete
  `state_dict`, including non-trainable buffers.

Release metadata records both values. The second value must not be described as
the number of trainable parameters.
