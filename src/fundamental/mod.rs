use std::cell::RefCell;
use std::rc::{Rc, Weak};

pub mod autograd;
pub mod unit;
pub mod op;

use unit::_Unit;
pub type Unit = Rc<RefCell<_Unit>>;
pub type UnitRef = Weak<RefCell<_Unit>>;