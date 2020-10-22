extern crate sdl2;
extern crate vulkano;
extern crate vulkano_shaders;
extern crate nalgebra as na;
extern crate num_traits;

mod shaders;
mod sendable;

use vulkano::instance::{RawInstanceExtensions, Instance, PhysicalDevice};
use std::ffi::CString;
use vulkano::VulkanObject;
use vulkano::swapchain;
use vulkano::swapchain::{Surface, Swapchain, SurfaceTransform, PresentMode, FullscreenExclusive, ColorSpace, AcquireError, SwapchainCreationError};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage, CpuBufferPool};
use std::sync::Arc;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::sync;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::framebuffer::{Subpass, RenderPassAbstract, FramebufferAbstract, Framebuffer};
use vulkano::command_buffer::{DynamicState, AutoCommandBufferBuilder};
use sdl2::video::{WindowContext};
use vulkano::pipeline::viewport::Viewport;
use vulkano::sync::{GpuFuture, FlushError};
use std::rc::Rc;
use crate::sendable::Sendable;
use std::time::Instant;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use na::{Matrix4, Perspective3};
use std::f32::consts::PI;


#[derive(Default, Copy, Clone)]
struct Vertex {
    position: [f32; 4],
    color: [f32; 4],
}

impl Vertex {
    fn new(data: [f32; 8]) -> Vertex {
        Vertex {
            position: [data[0], data[1], data[2], data[3]],
            color: [data[4], data[5], data[6], data[7]],
        }
    }
}

vulkano::impl_vertex!(Vertex, position, color);

fn main() {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem.window("Window", 800, 600)
        .resizable()
        .vulkan()
        .build()
        .unwrap();

    let instance_extensions = window.vulkan_instance_extensions().unwrap();
    let raw_instance_extensions = RawInstanceExtensions::new(instance_extensions.iter().map(
        |&v| CString::new(v).unwrap()
    ));

    let instance = Instance::new(None, raw_instance_extensions, None).unwrap();

    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");

    println!(
        "Using device: {} (type{:?})",
        physical.name(),
        physical.ty()
    );

    let surface_handle = window.vulkan_create_surface(instance.internal_object()).unwrap();

    let window_context = Sendable::new(window.context());

    let surface = Arc::new(unsafe {
        Surface::from_raw_surface(instance.clone(), surface_handle, window_context)
    });

    let queue_family = physical.queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .expect("couldn't find a graphycal queue family");

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (device, mut queues) = {
        Device::new(physical, physical.supported_features(), &device_ext,
                    [(queue_family, 0.5)].iter().cloned()).expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();

        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        let format = caps.supported_formats[0].0;

        let dimensions: [u32; 2] = [window.size().0, window.size().1];

        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            ImageUsage::color_attachment(),
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            FullscreenExclusive::Default,
            true,
            ColorSpace::SrgbNonLinear,
        ).unwrap()
    };

    let vertex_data: Vec<Vertex> = Vec::from([
        Vertex::new([0.25,  0.25, 0.75, 1.0, 0.0, 0.0, 1.0, 1.0]),
        Vertex::new([0.25, -0.25, 0.75, 1.0, 0.0, 0.0, 1.0, 1.0,]),
        Vertex::new([-0.25,  0.25, 0.75, 1.0, 0.0, 0.0, 1.0, 1.0,]),

        Vertex::new([0.25, -0.25, 0.75, 1.0, 0.0, 0.0, 1.0, 1.0]),
        Vertex::new([-0.25, -0.25, 0.75, 1.0, 0.0, 0.0, 1.0, 1.0,]),
        Vertex::new([-0.25,  0.25, 0.75, 1.0, 0.0, 0.0, 1.0, 1.0,]),

        Vertex::new([0.25,  0.25, -0.75, 1.0, 0.8, 0.8, 0.8, 1.0,]),
        Vertex::new([-0.25,  0.25, -0.75, 1.0, 0.8, 0.8, 0.8, 1.0,]),
        Vertex::new([0.25, -0.25, -0.75, 1.0, 0.8, 0.8, 0.8, 1.0,]),

        Vertex::new([0.25, -0.25, -0.75, 1.0, 0.8, 0.8, 0.8, 1.0,]),
        Vertex::new([-0.25,  0.25, -0.75, 1.0, 0.8, 0.8, 0.8, 1.0,]),
        Vertex::new([-0.25, -0.25, -0.75, 1.0, 0.8, 0.8, 0.8, 1.0]),

        Vertex::new([-0.25,  0.25,  0.75, 1.0, 0.0, 1.0, 0.0, 1.0,]),
        Vertex::new([-0.25, -0.25,  0.75, 1.0, 0.0, 1.0, 0.0, 1.0,]),
        Vertex::new([-0.25, -0.25, -0.75, 1.0, 0.0, 1.0, 0.0, 1.0]),

        Vertex::new([-0.25,  0.25,  0.75, 1.0, 0.0, 1.0, 0.0, 1.0,]),
        Vertex::new([-0.25, -0.25, -0.75, 1.0, 0.0, 1.0, 0.0, 1.0,]),
        Vertex::new([-0.25,  0.25, -0.75, 1.0, 0.0, 1.0, 0.0, 1.0,]),

        Vertex::new([0.25,  0.25,  0.75, 1.0, 0.5, 0.5, 0.0, 1.0,]),
        Vertex::new([0.25, -0.25, -0.75, 1.0, 0.5, 0.5, 0.0, 1.0]),
        Vertex::new([0.25, -0.25,  0.75, 1.0, 0.5, 0.5, 0.0, 1.0,]),

        Vertex::new([0.25,  0.25,  0.75, 1.0, 0.5, 0.5, 0.0, 1.0,]),
        Vertex::new([0.25,  0.25, -0.75, 1.0, 0.5, 0.5, 0.0, 1.0,]),
        Vertex::new([0.25, -0.25, -0.75, 1.0, 0.5, 0.5, 0.0, 1.0,]),

        Vertex::new([0.25,  0.25, -0.75, 1.0, 1.0, 0.0, 0.0, 1.0,]),
        Vertex::new([0.25,  0.25,  0.75, 1.0, 1.0, 0.0, 0.0, 1.0,]),
        Vertex::new([-0.25,  0.25,  0.75, 1.0, 1.0, 0.0, 0.0, 1.0,]),

        Vertex::new([0.25,  0.25, -0.75, 1.0, 1.0, 0.0, 0.0, 1.0,]),
        Vertex::new([-0.25,  0.25,  0.75, 1.0, 1.0, 0.0, 0.0, 1.0,]),
        Vertex::new([-0.25,  0.25, -0.75, 1.0, 1.0, 0.0, 0.0, 1.0,]),

        Vertex::new([0.25, -0.25, -0.75, 1.0, 0.0, 1.0, 1.0, 1.0,]),
        Vertex::new([-0.25, -0.25,  0.75, 1.0, 0.0, 1.0, 1.0, 1.0,]),
        Vertex::new([0.25, -0.25,  0.75, 1.0, 0.0, 1.0, 1.0, 1.0,]),

        Vertex::new([0.25, -0.25, -0.75, 1.0, 0.0, 1.0, 1.0, 1.0,]),
        Vertex::new([-0.25, -0.25, -0.75, 1.0, 0.0, 1.0, 1.0, 1.0,]),
        Vertex::new([-0.25, -0.25,  0.75, 1.0, 0.0, 1.0, 1.0, 1.0])
    ]);

    let vs = shaders::vs::Shader::load(device.clone()).unwrap();
    let fs = shaders::fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
            device.clone(),
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
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .cull_mode_back()
            .front_face_clockwise()
            .build(device.clone())
            .unwrap()
    );

    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: None,
        scissors: None,
        compare_mask: None,
        write_mask: None,
        reference: None
    };

    let mut framebuffers = window_size_dependent_setup(&images[..], render_pass.clone(), &mut dynamic_state);

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let mut event_pump = sdl_context.event_pump().unwrap();

    let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,
                                                       vertex_data.iter().cloned()).unwrap();

    let uniform_buffer = CpuBufferPool::<shaders::vs::ty::Data>::new(device.clone(), BufferUsage::all());

    let start_time = Instant::now();

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                    break 'running
                },
                Event::Window { win_event: WindowEvent::Resized(_, _), .. } => {
                    recreate_swapchain = true;
                },
                Event::MouseButtonDown { x, y, .. } => {
                    println!("Click at {:?}, {:?}", x, y);
                }
                _ => {}
            }
        }

        previous_frame_end.as_mut().unwrap().cleanup_finished();

        if recreate_swapchain {
            let dimensions: [u32; 2] = [window.size().0, window.size().1];
            let (new_swapchain, new_images) =
                match swapchain.recreate_with_dimensions(dimensions) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::UnsupportedDimensions) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                };

            swapchain = new_swapchain;

            framebuffers = window_size_dependent_setup(
                &new_images,
                render_pass.clone(),
                &mut dynamic_state
            );

            recreate_swapchain = false;
        }

        let (image_num, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };

        if suboptimal {
            recreate_swapchain = true;
        }

        let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into()];

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            device.clone(),
            queue.family(),
        ).unwrap();

        let uniform_buffer_subbuffer = {
            let loop_duration: f32 = 5.0;
            let scale: f32 = std::f64::consts::PI as f32 * 2.0 / loop_duration;

            let elapsed_time: f32 = start_time.elapsed().as_secs() as f32;
            let current_time_loop = elapsed_time % loop_duration;

            let x_offset = (current_time_loop * scale).cos() * 0.5;
            let y_offset = (current_time_loop * scale).sin() * 0.5;

            let perspective: Matrix4<f32> = Perspective3::new((window.size().0 / window.size().1) as f32, PI / 2.0, 0.5, 3.0).into_inner();
            let vulkan_inverted: Matrix4<f32> = Matrix4::new(
                1.0, 0.0, 0.0, 0.0,
                0.0, -1.0, 0.0, 0.0,
                0.0, 0.0, 0.5, 0.0,
                0.0, 0.0, 0.5, 1.0,
            );

            let uniform_data = shaders::vs::ty::Data {
                offset: [x_offset, y_offset],
                perspectiveMatrix: (vulkan_inverted).into(),
            };

            uniform_buffer.next(uniform_data).unwrap()
        };

        let layout = pipeline.descriptor_set_layout(0).unwrap();
        let set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(uniform_buffer_subbuffer)
                .unwrap()
                .build()
                .unwrap(),
        );

        builder
            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
            .unwrap()
            .draw(
                pipeline.clone(),
                &dynamic_state,
                vertex_buffer.clone(),
                set.clone(),
                (),
            )
            .unwrap()
            .end_render_pass()
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
        }

        ::std::thread::sleep(::std::time::Duration::new(0, 1_000_000_000u32 / 60));
    }
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Sendable<Rc<WindowContext>>>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone())
                .unwrap()
                .build()
                .unwrap(),
        ) as Arc<dyn FramebufferAbstract + Send + Sync>
    })
        .collect::<Vec<_>>()
}
