use sqlite::{open, Connection};

pub struct BoardDB {
    conn: Connection,
}

impl BoardDB {
    pub fn new(s: &str) -> Self {
        let conn = open(s).unwrap();

        let query = "
            create table if not exists board_record (
                att integer,
                def integer,
                flag integer,
                val real
            )
        ";

        conn.execute(query).unwrap();
        return BoardDB { conn: conn };
    }

    pub fn add(&self, att: u64, def: u64, flag: i32, val: f32) {
        let query = format!(
            "
                insert into board_record(att, def, flag, val)
                values({}, {}, {}, {})",
            att as i64, def as i64, flag, val
        );

        self.conn.execute(query).unwrap();
    }

    pub fn get(&self) {
        let query = "SELECT * FROM board_record";

        self.conn
            .iterate(query, |pairs| {
                for &(name, value) in pairs.iter() {
                    println!("{} = {}", name, value.unwrap());
                }
                true
            })
            .unwrap();
    }
}
