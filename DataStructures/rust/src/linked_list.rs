pub struct LinkedList<T> {
    pub val: Option<T>,
    pub next: Option<Box<LinkedList<T>>>,
}

impl LinkedList<i32> {
    /// # Creates an empty LinkedList that may hold i32 values
    /// 
    /// # Example
    /// ```
    /// use data_structures::linked_list;
    /// let list = linked_list::LinkedList::<i32>::new();
    /// ```
    pub fn new() -> LinkedList<i32> {
        LinkedList {
            val: None,
            next: None,
        }
    }
}
/*
#[cfg(test)]
mod tests {
    user super::*;

    #[test]
    fn test_linked_list_push_left() {
        let mut list: LinkedList<i32> = LinkedList::<i32>::new();
        list.push_left(1);
        list.push_left(2);
        list.push_left(3);
        list.push_left(4);
        assert_eq!(list.collect(), vec![4, 3, 2, 1]);
    }
}*/