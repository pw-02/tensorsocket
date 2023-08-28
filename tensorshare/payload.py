from torch import Tensor
from torch.multiprocessing.reductions import rebuild_cuda_tensor


class TensorPayload:
    def __init__(self, tensor: Tensor | tuple):
        """Tensor sharing payload

        Args:
            tensor (Tensor | tuple): Source tensor or payload
        """
        if isinstance(tensor, Tensor):
            self.payload = self._from_tensor(tensor)
            self._tensor = tensor
        else:
            self.payload = tensor
            self._tensor = rebuild_cuda_tensor(Tensor, **self.payload)

    def _from_tensor(self, tensor: Tensor) -> tuple:
        #storage = tensor.untyped_storage()
        storage = tensor._typed_storage()
        (
            storage_device,
            storage_handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_cuda_()

        payload = {
            "dtype": tensor.dtype,
            "tensor_size": tuple(tensor.size()),
            "tensor_stride": tensor.stride(),
            "tensor_offset": tensor.storage_offset(),
            "storage_cls": type(storage),
            "storage_device": storage_device,
            "storage_handle": storage_handle,
            "storage_size_bytes": int(storage_size_bytes),
            "storage_offset_bytes": storage_offset_bytes,
            "requires_grad": False,
            "ref_counter_handle": ref_counter_handle,
            "ref_counter_offset": ref_counter_offset,
            "event_handle": event_handle,
            "event_sync_required": event_sync_required,
        }
        return payload

    def __reduce__(self):
        return (
            self.__class__,
            (self.payload,),
        )

    @property
    def tensor(self):
        return self._tensor
