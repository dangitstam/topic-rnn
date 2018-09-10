
from typing import Callable, Optional, Tuple, Union, Dict

import torch
from allennlp.nn.util import (get_lengths_from_binary_sequence_mask,
                              sort_batch_by_length)
from torch.nn.utils.rnn import (PackedSequence, pack_padded_sequence,
                                pad_packed_sequence)

RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]  # pylint: disable=invalid-name
RnnStateStorage = Tuple[torch.Tensor, ...]  # pylint: disable=invalid-name


def sort_and_run_forward(module: Callable[[PackedSequence, Optional[RnnState]],
                                          Tuple[Union[PackedSequence, torch.Tensor], RnnState]],
                         inputs: torch.Tensor,
                         mask: torch.Tensor,
                         hidden_state: Optional[RnnState] = None):
    """
    Adapted from AllenNLP's `sort_and_run_forward` from `encoder_base.py`

    This function exists because Pytorch RNNs require that their inputs be sorted
    before being passed as input. As all of our Seq2xxxEncoders use this functionality,
    it is provided in a base class. This method can be called on any module which
    takes as input a ``PackedSequence`` and some ``hidden_state``, which can either be a
    tuple of tensors or a tensor.
    As all of our Seq2xxxEncoders have different return types, we return `sorted`
    outputs from the module, which is called directly. Additionally, we return the
    indices into the batch dimension required to restore the tensor to it's correct,
    unsorted order and the number of valid batch elements (i.e the number of elements
    in the batch which are not completely masked). This un-sorting and re-padding
    of the module outputs is left to the subclasses because their outputs have different
    types and handling them smoothly here is difficult.
    Parameters
    ----------
    module : ``Callable[[PackedSequence, Optional[RnnState]],
                        Tuple[Union[PackedSequence, torch.Tensor], RnnState]]``, required.
        A function to run on the inputs. In most cases, this is a ``torch.nn.Module``.
    inputs : ``torch.Tensor``, required.
        A tensor of shape ``(batch_size, sequence_length, embedding_size)`` representing
        the inputs to the Encoder.
    mask : ``torch.Tensor``, required.
        A tensor of shape ``(batch_size, sequence_length)``, representing masked and
        non-masked elements of the sequence for each element in the batch.
    hidden_state : ``Optional[RnnState]``, (default = None).
        A single tensor of shape (num_layers, batch_size, hidden_size) representing the
        state of an RNN with or a tuple of
        tensors of shapes (num_layers, batch_size, hidden_size) and
        (num_layers, batch_size, memory_size), representing the hidden state and memory
        state of an LSTM-like RNN.
    Returns
    -------
    module_output : ``Union[torch.Tensor, PackedSequence]``.
        A Tensor or PackedSequence representing the output of the Pytorch Module.
        The batch size dimension will be equal to ``num_valid``, as sequences of zero
        length are clipped off before the module is called, as Pytorch cannot handle
        zero length sequences.
    final_states : ``Optional[RnnState]``
        A Tensor representing the hidden state of the Pytorch Module. This can either
        be a single tensor of shape (num_layers, num_valid, hidden_size), for instance in
        the case of a GRU, or a tuple of tensors, such as those required for an LSTM.
    restoration_indices : ``torch.LongTensor``
        A tensor of shape ``(batch_size,)``, describing the re-indexing required to transform
        the outputs back to their original batch order.
    """
    # In some circumstances you may have sequences of zero length. ``pack_padded_sequence``
    # requires all sequence lengths to be > 0, so remove sequences of zero length before
    # calling self._module, then fill with zeros.

    # First count how many sequences are empty.
    batch_size = mask.size(0)
    num_valid = torch.sum(mask[:, 0]).int().item()

    sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
    sorted_inputs, sorted_sequence_lengths, restoration_indices, sorting_indices =\
        sort_batch_by_length(inputs, sequence_lengths)

    # Now create a PackedSequence with only the non-empty, sorted sequences.
    packed_sequence_input = pack_padded_sequence(sorted_inputs[:num_valid, :, :],
                                                 sorted_sequence_lengths[:num_valid].data.tolist(),
                                                 batch_first=True)
    # Prepare the initial states.
    if hidden_state is None:
        initial_states = hidden_state
    elif isinstance(hidden_state, tuple):
        initial_states = [state.index_select(1, sorting_indices)[:, :num_valid, :].contiguous()
                          for state in hidden_state]
    else:
        initial_states = hidden_state.index_select(1, sorting_indices)[:, :num_valid, :].contiguous()


    # Actually call the module on the sorted PackedSequence.
    module_output, final_states = module(packed_sequence_input, initial_states)

    return module_output, final_states, restoration_indices


# Assume all RNNs using this is stateful.
def rnn_forward(module: Callable[[PackedSequence, Optional[RnnState]],
                                 Tuple[Union[PackedSequence, torch.Tensor], RnnState]],  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.Tensor,
                hidden_state: torch.Tensor = None) -> torch.Tensor:
    """
    This version of forward for Seq2Seq Encoders also returns the final hidden states.
    """

    if mask is None:
        return module(inputs, hidden_state)[0]

    batch_size, total_sequence_length = mask.size()

    packed_sequence_output, final_states, restoration_indices = \
        sort_and_run_forward(module, inputs, mask, hidden_state)

    unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)

    num_valid = unpacked_sequence_tensor.size(0)

    # Add back invalid rows.
    if num_valid < batch_size:
        _, length, output_dim = unpacked_sequence_tensor.size()
        zeros = unpacked_sequence_tensor.new_zeros(batch_size - num_valid, length, output_dim)
        unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 0)

        # We also do this to the final hidden states.
        zeros = final_states.new_zeros(final_states.size(0),
                                       batch_size - num_valid,
                                       final_states.size(-1))
        final_states = torch.cat([final_states, zeros], 1)

    # It's possible to need to pass sequences which are padded to longer than the
    # max length of the sequence to a Seq2SeqEncoder. However, packing and unpacking
    # the sequences mean that the returned tensor won't include these dimensions, because
    # the RNN did not need to process them. We add them back on in the form of zeros here.
    sequence_length_difference = total_sequence_length - unpacked_sequence_tensor.size(1)
    if sequence_length_difference > 0:
        zeros = unpacked_sequence_tensor.new_zeros(batch_size,
                                                   sequence_length_difference,
                                                   unpacked_sequence_tensor.size(-1))
        unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 1)

    # Restore the original indices and return the sequence.
    return (unpacked_sequence_tensor.index_select(0, restoration_indices),
            final_states.index_select(1, restoration_indices))

def description_from_metrics(metrics: Dict[str, float],
                             batch_iteration: int = None) -> str:
    description = ', '.join(["%s: %.4f" % (name, value)
                             for name, value in metrics.items()
                             if not name.startswith("_")]) + " ||"

    if batch_iteration:
        description = ("Batch #: %.4d " % batch_iteration) + description

    return description
