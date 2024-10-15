from torch import Tensor
from torch.multiprocessing.reductions import rebuild_tensor, rebuild_cuda_tensor


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

            if "storage_cls" in self.payload:
                try:
                    self._tensor = rebuild_cuda_tensor(Tensor, **self.payload)
                except RuntimeError as e:
                    self._tensor = tensor
            else:
                self._tensor = rebuild_tensor(
                    tensor["cls"], tensor["storage"], tensor["metadata"]
                )

    def _from_tensor(self, tensor: Tensor) -> tuple:
        # storage = tensor.untyped_storage()
        storage = tensor._typed_storage()

        if storage.is_cuda:
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

            return {
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

        storage.share_memory_()
        metadata = (
            tensor.storage_offset(),
            tensor.size(),
            tensor.stride(),
            tensor.requires_grad,
        )
        return {
            "storage": storage,
            "cls": type(storage),
            "metadata": metadata,
        }

    def __reduce__(self):
        return (
            self.__class__,
            (self.payload,),
        )

    @property
    def tensor(self):
        return self._tensor
