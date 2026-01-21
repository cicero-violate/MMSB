// src/scheduler.rs
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;
use tokio::time::sleep;

/// Scheduler trait — wake coordination only.
/// No authority. No state. No MMSB access.
pub trait Scheduler: Send + Sync {
    fn wait_for_tick(&self) -> Pin<Box<dyn Future<Output = ()> + Send>>;
    fn wake(&self);
    fn wait(&self) -> Pin<Box<dyn Future<Output = ()> + Send>>;
}

/// Default runtime scheduler implementation.
pub struct RuntimeScheduler {
    tick_interval: Duration,
    notify: Arc<Notify>,
}

impl RuntimeScheduler {
    pub fn new(tick_interval: Duration) -> Self {
        Self {
            tick_interval,
            notify: Arc::new(Notify::new()),
        }
    }
}

impl Scheduler for RuntimeScheduler {
    fn wait_for_tick(&self) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        let interval = self.tick_interval;
        Box::pin(async move {
            sleep(interval).await;
        })
    }

    fn wake(&self) {
        self.notify.notify_waiters();
    }

    fn wait(&self) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        let interval = self.tick_interval;
        let notify = Arc::clone(&self.notify);
        Box::pin(async move {
            tokio::select! {
                _ = sleep(interval) => {}
                _ = notify.notified() => {}
            }
        })
    }
}

impl Default for RuntimeScheduler {
    fn default() -> Self {
        Self::new(Duration::from_millis(10))
    }
}
