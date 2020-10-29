use vulkano::instance::{RawInstanceExtensions, Instance, PhysicalDevice};
use std::ffi::CString;
use vulkano::device::{DeviceExtensions, Device, Queue};
use std::sync::Arc;
use sdl2::video::{WindowContext, Window};
use std::rc::Rc;
use crate::sendable::Sendable;
use vulkano::swapchain::Surface;
use vulkano::VulkanObject;

pub struct Renderer {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub surface: Arc<Surface<Sendable<Rc<WindowContext>>>>,
}

#[derive(Debug)]
pub enum Error {
    InitError,
}

impl Renderer {
    pub fn init_vulkan(window: &Window) -> Result<Renderer, Error> {
        let instance_extensions = window.vulkan_instance_extensions().unwrap();

        let window_context = Sendable::new(window.context());

        let raw_instance_extensions = RawInstanceExtensions::new(instance_extensions.iter().map(
            |&v| CString::new(v).unwrap()
        ));

        let instance = Instance::new(None, raw_instance_extensions, None).unwrap();
        let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

        let surface_handle = window.vulkan_create_surface(instance.internal_object()).unwrap();
        let surface = Arc::new(unsafe {
            Surface::from_raw_surface(instance.clone(), surface_handle, window_context)
        });

        let queue_family = physical.queue_families()
            .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false)).unwrap();

        let device_ext = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };

        let (device, mut queues) = {
            Device::new(physical, physical.supported_features(), &device_ext,
                        [(queue_family, 0.5)].iter().cloned()).unwrap()
        };

        let queue = queues.next().unwrap();

        Ok(Renderer {
            device,
            queue,
            surface,
        })
    }
}