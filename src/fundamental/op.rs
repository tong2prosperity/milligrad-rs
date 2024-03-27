use super::*;
use super::unit::{new_unit, Operation};



pub fn add(l: &Unit, r: &Unit) -> Unit {
    let mut output = _Unit::new(l.borrow().data + r.borrow().data);
    let (l_rc, r_rc) = (Rc::downgrade(l), Rc::downgrade(r));
    output.operation = Some(Operation::Add(l_rc, r_rc));
    output.children.push(l.clone());
    output.children.push(r.clone());
    output.to_unit()
}

pub fn mul(l: &Unit, r: &Unit) -> Unit {
    let mut output = _Unit::new(l.borrow().data * r.borrow().data);
    let (l_rc, r_rc) = (Rc::downgrade(l), Rc::downgrade(r));
    output.operation = Some(Operation::Mul(l_rc, r_rc));
    output.children.push(l.clone());
    output.children.push(r.clone());
    output.to_unit()
}

pub fn tanh(x: &Unit) -> Unit {
    let mut output = _Unit::new(x.borrow().data.tanh());
    let x_rc = Rc::downgrade(x);
    output.operation = Some(Operation::Tanh(x_rc));
    output.children.push(x.clone());
    output.to_unit()
}


pub fn backward(u: &Unit) {
    u.borrow_mut().grad = 1.0;
    let  topo_n = rev_topological_sort_dfs(u);
    //topo_n.reverse();
    for bu in topo_n {
        bu.upgrade().unwrap().borrow_mut().self_back_propagation();
    }
}


use std::collections::{HashMap, HashSet, VecDeque};
fn topological_sort(graph: &HashMap<usize, Vec<usize>>) -> Option<Vec<usize>> {
    let mut in_degrees = HashMap::new();
    let mut queue = VecDeque::new();
    let mut sorted = Vec::new();
    // 计算每个顶点的入度
    for (&node, neighbors) in graph {
        in_degrees.entry(node).or_insert(0);
        for &neighbor in neighbors {
            *in_degrees.entry(neighbor).or_insert(0) += 1;
        }
    }
    // 将所有入度为0的顶点加入队列
    for (&node, &in_degree) in in_degrees.iter() {
        if in_degree == 0 {
            queue.push_back(node);
        }
    }
    // Kahn算法主循环
    while let Some(node) = queue.pop_front() {
        sorted.push(node);
        if let Some(neighbors) = graph.get(&node) {
            for &neighbor in neighbors {
                let entry = in_degrees.get_mut(&neighbor)?;
                *entry -= 1;
                if *entry == 0 {
                    queue.push_back(neighbor);
                }
            }
        }
    }
    // 检查是否存在环
    if sorted.len() == in_degrees.len() {
        Some(sorted)
    } else {
        None
    }
}

fn rev_topological_sort_dfs(u:&Unit) -> Vec<UnitRef> {
    let mut visited: HashSet<usize> = HashSet::new();
    let mut sorted = Vec::new();
    let mut stack = VecDeque::new();


    stack.push_back(u.clone());

    while let Some(current) = stack.pop_back() {
        if visited.contains(&current.borrow().id()) {
            continue;
        }
        visited.insert(current.borrow().id());
        for child in current.borrow().children.iter() {
            stack.push_back(child.clone());
        }
        sorted.push(Rc::downgrade(&current));
    }

    //sorted.reverse();
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::unit::Operation;


    #[test]
    fn test_back_propagation() {
        let a = new_unit(1.0);
        let b = new_unit(2.0);
        let mut c = add(&a, &b);
        c.borrow_mut().grad = 1.0;
        c.borrow_mut().self_back_propagation();
        println!("{:?}", c.borrow());
        assert_eq!(a.borrow().grad, 1.0);
        assert_eq!(b.borrow().grad, 1.0);
    }

    use std::collections::{HashMap, HashSet, VecDeque};

    #[test]
    fn test_topo() {
        let graph = vec![
            (2, vec![3, 0, 1]),
            (3, vec![1]),
            (0, vec![1]),
            (1, vec![4]),
        ].iter().cloned().collect::<HashMap<_, _>>();
        match topological_sort(&graph) {
            Some(sorted) => println!("Topologically sorted: {:?}", sorted),
            None => println!("No topological sort exists due to cycles in the graph."),
        }
    }

    #[test]
    fn test_topo_dfs() {
        let a = new_unit(1.0);
        let b = new_unit(2.0);
        let c = add(&a, &b); // 3
        let d = new_unit(4.0); // 2
        let e = tanh(&c); // 0.99505475
        let f = add(&d, &e); // 2.99505475
        let mut sorted = rev_topological_sort_dfs(&f);
        sorted.reverse();
        for  s in sorted.iter() {
            println!("{:}", s.upgrade().unwrap().borrow());
        }

        backward(&f);

        for ref s in sorted {
            println!("{:}", s.upgrade().unwrap().borrow());
        }
    }

}
