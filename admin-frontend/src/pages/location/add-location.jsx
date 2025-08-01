import React from 'react';
import PageHeader from '@/components/shared/pageHeader/PageHeader';
import Footer from '@/components/shared/Footer';
import AddLocationForm from '@/components/location/AddLocationForm';

const AddLocation = () => {
    return (
        <>
            <PageHeader >
            </PageHeader>
            <div className='main-content'>
                <div className='row'>
                    <AddLocationForm title={"Add Location"} />
                </div>
            </div>
            <Footer />
        </>
    );
}

export default AddLocation;