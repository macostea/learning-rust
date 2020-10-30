use vulkano::instance::{RawInstanceExtensions, Instance, PhysicalDevice};
use std::ffi::CString;
use vulkano::device::{DeviceExtensions, Device, Queue};
use std::sync::Arc;
use sdl2::video::{WindowContext, Window};
use std::rc::Rc;
use crate::sendable::Sendable;
use vulkano::swapchain::{Surface, Swapchain, SurfaceTransform, PresentMode, FullscreenExclusive, ColorSpace};
use vulkano::VulkanObject;
use vulkano::image::{ImageUsage, SwapchainImage};
use std::borrow::Borrow;
use vulkano::framebuffer::{RenderPass, RenderPassAbstract};

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
        let window = window.clone();

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

    pub fn create_swapchain(&self, window: &Window) -> (Arc<Swapchain<Sendable<Rc<WindowContext>>>>, Vec<Arc<SwapchainImage<Sendable<Rc<WindowContext>>>>>) {
        let caps = self.surface.clone().capabilities(self.device.clone().physical_device()).unwrap();

        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        let format = caps.supported_formats[0].0;

        let dimensions: [u32; 2] = [window.size().0, window.size().1];

        Swapchain::new(
            self.device.clone(),
            self.surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            ImageUsage::color_attachment(),
            &self.queue.clone(),
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            FullscreenExclusive::Default,
            true,
            ColorSpace::SrgbNonLinear,
        ).unwrap()
    }

    pub fn create_renderpass(&self, swapchain: Arc<Swapchain<Sendable<Rc<WindowContext>>>>) -> Arc<dyn RenderPassAbstract + Send + Sync> {
        Arc::new(
            vulkano::single_pass_renderpass!(
            self.device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).unwrap(),
        )
    }
}
